import torch
import torch.optim.lr_scheduler as lr_scheduler
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic
from ..utils.attention import SelfAttention, CrossAttention


class PPOPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr
        self.lr_min = 1e-5
        self.attention_feature_dim = 7

        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space

        self.self_attention = SelfAttention(self.attention_feature_dim)
        self.cross_attention = CrossAttention(self.attention_feature_dim)

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.cent_obs_space, self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()},
            {'params': self.self_attention.parameters()},
            {'params': self.cross_attention.parameters()}
        ], lr=self.lr)

        # 设置固定更新步骤降低一次学习率，每次降低都乘以gamma
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=150, gamma=0.90)

    def generate_attention_feature(self, ally_attention_n, enemy_attention_n):
        # 计算混合特征,需要输入己方特征
        ally_attention_output = self.self_attention(ally_attention_n)
        ally_fused_vector = ally_attention_output.sum(dim=0)  # [dim]
        ally_fused_vector = ally_fused_vector.unsqueeze(0)  # [1, dim]

        # 根据己方融合特征计算出敌方的融合特征,需要输入己方融合特征与敌方特征
        enemy_fused_vector, attention_fused_weight = self.cross_attention(ally_fused_vector,
                                                                          enemy_attention_n,
                                                                          enemy_attention_n)  # [1, d]

        return enemy_fused_vector, attention_fused_weight

    def get_actions(self, cent_obs, obs, ally_attention_n, enemy_attention_n, rnn_states_actor, rnn_states_critic,
                    masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        enemy_fused_vector, attention_fused_weight = self.generate_attention_feature(ally_attention_n, enemy_attention_n)
        attention_fused_weight = attention_fused_weight.reshape((-1, 1))

        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)

        return values, actions, attention_fused_weight, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, ally_attention_n, enemy_attention_n, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, ally_attention_n, enemy_attention_n, rnn_states_actor, rnn_states_critic,
                         action, masks, active_masks=None):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, ally_attention_n, enemy_attention_n, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        enemy_fused_vector, attention_fused_weight = self.generate_attention_feature(ally_attention_n, enemy_attention_n)
        attention_fused_weight = attention_fused_weight.reshape((-1, 1))

        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)

        return actions, attention_fused_weight, rnn_states_actor

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
