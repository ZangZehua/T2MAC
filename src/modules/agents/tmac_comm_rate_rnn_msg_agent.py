import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnMsgAgent(nn.Module):
    """
    input shape: [batch_size, in_feature]
    output shape: [batch_size, n_actions]
    hidden state shape: [batch_size, hidden_dim]
    """

    def __init__(self, input_dim, args):
        super().__init__()

        self.args = args
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc_value = nn.Linear(args.rnn_hidden_dim, args.n_value)
        self.fc_key = nn.Linear(args.rnn_hidden_dim, args.n_key)
        self.fc_query = nn.Linear(args.rnn_hidden_dim, args.n_query)

        self.fc_attn = nn.Linear(args.n_query + args.n_key * args.n_agents, args.n_agents)

        self.fc_attn_combine = nn.Linear(args.n_value + args.rnn_hidden_dim, args.rnn_hidden_dim)

        # used when ablate 'shortcut' connection
        # self.fc_attn_combine = nn.Linear(args.n_value, args.rnn_hidden_dim)

    def forward(self, x, hidden):
        """
        hidden state: [batch_size, n_agents, hidden_dim]
        q_without_communication
        """
        x = F.relu(self.fc1(x))
        h_in = hidden.view(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        h_out = h_out.view(-1, self.args.n_agents, self.args.rnn_hidden_dim)
        return h_out

    def q_without_communication(self, h_out):
        q_without_comm = self.fc2(h_out)
        return q_without_comm

    def communicate(self, hidden):
        """
        input: hidden [batch_size, n_agents, hidden_dim]
        output: key, value, signature
        """
        key = self.fc_key(hidden)
        value = self.fc_value(hidden)
        query = self.fc_query(hidden)

        return key, value, query

    def aggregate(self, query, key, value, hidden, send_target):
        """
        query: [batch_size, n_agents, n_query]
        key: [batch_size, n_agents, n_key]
        value: [batch_size, n_agents, n_value]
        """
        n_agents = self.args.n_agents
        _key = torch.cat([key[:, i, :] for i in range(n_agents)], dim=-1).unsqueeze(1).repeat(1, n_agents, 1)
        query_key = torch.cat([query, _key], dim=-1)  # [batch_size, n_agents, n_query + n_agents*n_key]

        # attention weights
        attn_weights = F.softmax(self.fc_attn(query_key), dim=-1)  # [batch_size, n_agents, n_agents]

        # attentional value
        attn_applied = torch.bmm(attn_weights, value)  # [batch_size, n_agents, n_value]

        # shortcut connection: combine with agent's own hidden
        attn_combined = torch.cat([attn_applied, hidden], dim=-1)

        # used when ablate 'shortcut' connection
        # attn_combined = attn_applied

        attn_combined = F.relu(self.fc_attn_combine(attn_combined))

        # mlp, output Q
        q = self.fc2(attn_combined)  # [batch_size, n_agents, n_actions]
        evidence = torch.clamp(q, 0, torch.inf)
        # print("----------")
        # print(evidence, type(evidence), evidence.dtype)
        received_evidence, ori_u, com_u = self.combine_message(evidence, send_target)
        # print(evidence, type(evidence), evidence.dtype)
        evidence = evidence + received_evidence
        # print(evidence, type(evidence), evidence.dtype)
        return evidence, ori_u, com_u

    # @torch.no_grad()
    def combine_message(self, evidence, send_target):
        batch_evidence = evidence.clone().detach()
        alpha = batch_evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        batch_belief = (batch_evidence / (S.expand(batch_evidence.shape))).transpose(0, 1)
        batch_uncertainty = (self.args.n_actions / S).transpose(0, 1)
        received_belief = torch.zeros((batch_evidence.shape[0], self.args.n_actions)).cuda()
        received_uncertainty = torch.ones((batch_evidence.shape[0], 1)).cuda()
        for b, u in zip(batch_belief, batch_uncertainty):
            received_belief, received_uncertainty = self.combine(received_belief, received_uncertainty, b, u)
        all_evidence = self.belief_to_evidence(received_belief, received_uncertainty)
        received_evidence = all_evidence.unsqueeze(1).repeat(1, self.args.n_agents, 1)
        scale = (torch.max(batch_evidence) - torch.min(batch_evidence)) * self.args.comm_coef
        received_evidence = scale * (received_evidence - torch.min(received_evidence)) / (torch.max(
            received_evidence) - torch.min(received_evidence) + 0.01)
        combined_evidence = batch_evidence + received_evidence

        alpha = combined_evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        combined_uncertainty = (self.args.n_actions / S).transpose(0, 1)
        return combined_evidence, batch_uncertainty, combined_uncertainty

    # 还是高
    # @torch.no_grad()
    # def combine_message(self, evidence, send_target):
    #     batch_evidence = evidence.clone().detach()
    #     alpha = batch_evidence + 1
    #     S = torch.sum(alpha, dim=-1, keepdim=True)
    #     batch_belief = (batch_evidence / (S.expand(batch_evidence.shape))).transpose(0, 1)
    #     batch_uncertainty = (self.args.n_actions / S).transpose(0, 1)
    #     received_belief = torch.zeros((batch_evidence.shape[0], self.args.n_actions)).cuda()
    #     received_uncertainty = torch.ones((batch_evidence.shape[0], 1)).cuda()
    #     for b, u in zip(batch_belief, batch_uncertainty):
    #         received_belief, received_uncertainty = self.combine(received_belief, received_uncertainty, b, u)
    #     all_evidence = self.belief_to_evidence(received_belief, received_uncertainty)
    #     received_evidence = torch.tensor([]).cuda()
    #     for env_send_target, env_evidence, env_re_evidence in zip(send_target, batch_evidence, all_evidence):
    #         for agent_send_target in env_send_target:
    #             rc_e = env_re_evidence.clone()
    #             for target, e in zip(agent_send_target, env_evidence):
    #                 if target.item() == 0:
    #                     rc_e = rc_e - (e / self.args.n_agents)
    #             received_evidence = torch.cat((received_evidence, rc_e.unsqueeze(0)), dim=0)
    #     received_evidence = received_evidence.reshape(batch_evidence.shape[0], self.args.n_agents,
    #                                                   received_evidence.shape[-1])
    #     scale = (torch.max(batch_evidence) - torch.min(batch_evidence)) * self.args.comm_coef
    #     received_evidence = scale * (received_evidence - torch.min(received_evidence)) / (torch.max(
    #         received_evidence) - torch.min(received_evidence) + 0.01)
    #     combined_evidence = batch_evidence + received_evidence
    #
    #     alpha = combined_evidence + 1
    #     S = torch.sum(alpha, dim=-1, keepdim=True)
    #     combined_uncertainty = (self.args.n_actions / S).transpose(0, 1)
    #     return combined_evidence, batch_uncertainty, combined_uncertainty

    # 时间复杂度太高
    # @torch.no_grad()
    # def combine_message(self, evidence, send_target):
    #     batch_evidence = evidence.detach()
    #     batch_send_target = send_target.transpose(-2, -1)
    #     alpha = batch_evidence + 1
    #     S = torch.sum(alpha, dim=-1, keepdim=True)
    #     batch_belief = batch_evidence / (S.expand(batch_evidence.shape))
    #     batch_uncertainty = self.args.n_actions / S
    #     received_belief = torch.zeros((1, self.args.n_actions)).cuda()
    #     received_uncertainty = torch.ones((1, 1)).cuda()
    #     combined_belief = torch.tensor([]).cuda()
    #     combined_uncertainty = torch.tensor([]).cuda()
    #     count = 0
    #     s_time = datetime.datetime.now()
    #     for env_send_target, env_belief, env_uncertainty in zip(batch_send_target, batch_belief, batch_uncertainty):
    #         for agent_send_target in env_send_target:
    #             r_b = received_belief.clone()
    #             r_u = received_uncertainty.clone()
    #             for target, b, u in zip(agent_send_target, env_belief, env_uncertainty):
    #                 if target.item() == 1:
    #                     count += 1
    #                     r_b, r_u = self.combine(r_b, r_u, b.unsqueeze(0), u.unsqueeze(0))
    #             combined_belief = torch.cat((combined_belief, r_b), dim=0)
    #             combined_uncertainty = torch.cat((combined_uncertainty, r_u), dim=0)
    #     combined_belief = combined_belief.reshape(evidence.shape[0], self.args.n_agents, combined_belief.shape[-1])
    #     combined_uncertainty = combined_uncertainty.reshape(evidence.shape[0], self.args.n_agents, 1)
    #     combined_evidence = self.belief_to_evidence(combined_belief, combined_uncertainty)
    #     combined_evidence  = evidence + 0.1 * combined_evidence
    #     return combined_evidence

    def belief_to_evidence(self, belief, uncertainty):
        S_a = self.args.n_actions / uncertainty
        # calculate new e_k
        e_a = torch.mul(belief, S_a.expand(belief.shape))
        return e_a

    def combine(self, b0, u0, b1, u1):
        # b^0 @ b^(0+1)
        bb = torch.bmm(b0.view(-1, self.args.n_actions, 1), b1.view(-1, 1, self.args.n_actions))
        # b^0 * u^1
        uv1_expand = u1.expand(b0.shape)
        bu = torch.mul(b0, uv1_expand)
        # b^1 * u^0
        uv_expand = u0.expand(b0.shape)
        ub = torch.mul(b1, uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b0, b1) + bu + ub) / ((1 - C).view(-1, 1).expand(b0.shape))
        # calculate u^a
        u_a = torch.mul(u0, u1) / ((1 - C).view(-1, 1).expand(u0.shape))
        return b_a, u_a

    def init_hidden(self):
        # trick, create hidden state on same device
        # batch size: 1
        return self.fc1.weight.new_zeros(1, self.args.rnn_hidden_dim)
