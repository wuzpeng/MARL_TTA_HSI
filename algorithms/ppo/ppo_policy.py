import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


class PPOPolicy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.obs_space, self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values, _ = self.critic(obs, rnn_states_critic, masks)  # _是一个占位符，这里被忽略
        return values

    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        """
        将模型切换为训练模式
        """
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        """
        将模型设置为推断模式，在该模式下，生成输出而不进行梯度计算
        """
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        """
        得到一个与当前模型结构和参数相同的全新实例，而不会共享模型参数
        """
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
