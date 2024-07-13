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

    def aggregate(self, query, key, value, hidden):
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
        q = torch.clamp(q, 0, torch.inf)
        return q

    def init_hidden(self):
        # trick, create hidden state on same device
        # batch size: 1
        return self.fc1.weight.new_zeros(1, self.args.rnn_hidden_dim)
