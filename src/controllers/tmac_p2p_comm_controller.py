import torch
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import itertools


# This multi-agent controller shares parameters between agents
class VffacMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        param = itertools.chain(self.agent.fc_msg_1.parameters(), self.agent.fc_msg_2.parameters())
        self.msg_optim = torch.optim.Adam(param, lr=0.001)
        self.crit_fun = torch.nn.CrossEntropyLoss().cuda()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # logging.debug(agent_outputs)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    @torch.no_grad()
    def generate_send_target(self, send_prob):
        send_target = torch.where(send_prob > 0.75, 1, 0)
        if len(send_target.shape) == 2:
            send_target = send_target.unsqueeze(0)
        return send_target

    def forward(self, ep_batch, t, test_mode=False, counterfactual=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        if counterfactual:
            q_without_comm = self.agent.q_without_communication(self.hidden_states)

        agents_key, agents_value, agents_query = self.agent.communicate(self.hidden_states)

        send_prob = self.agent.generate_send_prob(agent_inputs)

        send_target = torch.clamp(self.generate_send_target(send_prob) - torch.eye(self.n_agents).cuda(), 0,
                                  1).int().detach()

        agents_out, u_err = self.agent.aggregate(agents_query, agents_key, agents_value,
                                                 self.hidden_states, send_target)  # [batch_size, n_agents, n_actions]

        if u_err.shape[0] == self.args.batch_size:
            true_label = self.generate_true_label(u_err, send_target)
            loss = self.crit_fun(true_label, send_prob)
            self.msg_optim.zero_grad()
            loss.backward()
            self.msg_optim.step()

        return (agents_out, q_without_comm) if counterfactual else agents_out

    def generate_true_label(self, u_err, send_target):
        u_err = u_err.repeat(1, 1, send_target.shape[-1])
        true_label = torch.where(u_err > self.args.min_uncer, 1, 0)
        return true_label.float()

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
