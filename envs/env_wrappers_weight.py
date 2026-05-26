"""
A simplified version from OpenAI Baselines code to work with gym.env parallelization.
"""
import os
import contextlib
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

# 对象序列化，方便在多进程间传递
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


# 用于启动多进程时临时清楚环境变量
@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
    Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


# 抽象类，一个抽象的异步，向量化环境，用于处理来自多个环境的数据
class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions, attention_fused_weight):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, actions, attention_fused_weight):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions, attention_fused_weight)
        return self.step_wait()


# VecEnv实现，顺序地运行多个环境，适用于调试和当只有一个环境时避免通信开销
class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(self.envs), env.observation_space, env.action_space)

        self.actions = None
        self.attention_weight = None
        self.num_agents = getattr(self.envs[0], "num_agents", 1)

    def step_async(self, actions, attention_fused_weight):
        self.actions = actions
        self.attention_weight = attention_fused_weight

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obss, rewards, dones, infos = map(list, zip(*results))
        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obss[i] = self.envs[i].reset()
            elif isinstance(done, (list, tuple, np.ndarray)):
                if np.all(done):
                    obss[i] = self.envs[i].reset()
            elif isinstance(done, dict):
                if np.all(list(done.values())):
                    obss[i] = self.envs[i].reset()
            else:
                raise NotImplementedError("Unexpected type of done!")
        self.actions = None
        return self._flatten(obss), self._flatten(rewards), self._flatten(dones), np.array(infos)

    def reset(self):
        obss = [env.reset() for env in self.envs]
        return self._flatten(obss)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode, filepath):
        if mode == 'txt':
            self.envs[0].render(mode, filepath)

    @classmethod
    def _flatten(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0

        if isinstance(v[0], dict):
            return {k: np.stack([v_[k] for v_ in v]) for k in v[0].keys()}
        else:
            return np.stack(v)


# 用于在子进程中维护环境实例，通过管道与父进程通信
def worker(remote: Connection, parent_remote: Connection, env_fn_wrappers):
    """Maintain an environment instance in subprocess,
    communicate with parent-process via multiprocessing.Pipe.

    Args:
        remote (Connection): used for current subprocess to send/receive data.
        parent_remote (Connection): used for mainprocess to send/receive data. [Need to be closed in subprocess!]
        env_fn_wrappers (method): functions to create gym.Env instance.
    """
    def step_env(env, action, attention_weight):
        obs, reward, done, info = env.step(action, attention_weight)
        if 'bool' in done.__class__.__name__:
            if done:
                obs = env.reset()
        elif isinstance(done, (list, tuple, np.ndarray)):
            if np.all(done):
                obs = env.reset()
        elif isinstance(done, dict):
            if np.all(list(done.values())):
                obs = env.reset()
        else:
            raise NotImplementedError("Unexpected type of done!")
        return obs, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action, attention_weight) for env, action, attention_weight in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space)))
            elif cmd == 'get_num_agents':
                remote.send(CloudpickleWrapper((getattr(envs[0], "num_agents", 1))))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


# 在子进程中并行运行多个环境，并通过管道与它们通信
class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, context='spawn', in_series=1):
        """
        Args:
            env_fns: iterable of callables - functions that create environments to run in subprocesses. Need to be cloud-pickleable
            context (str, optional): Defaults to 'spawn'.
            in_series (int, optional): number of environments to run in series in a single process. Defaults to 1.
                (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        # create Pipe connections to send/recv data from subprocesses,
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nremotes)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv().x
        super().__init__(nenvs, observation_space, action_space)

        self.remotes[0].send(('get_num_agents', None))
        self.num_agents = self.remotes[0].recv().x

    def step_async(self, actions, attention_weight):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        attention_weight = np.array_split(attention_weight, self.nremotes)
        for remote, action, attention_weight in zip(self.remotes, actions, attention_weight):
            remote.send(('step', action, attention_weight))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results)  # [[tuple] * in_series] * nremotes => [tuple] * nenvs
        self.waiting = False
        obss, rewards, dones, infos = zip(*results)
        return self._flatten(obss), self._flatten(rewards), self._flatten(dones), np.array(infos)

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obss = [remote.recv() for remote in self.remotes]
        obss = self._flatten_series(obss)
        return self._flatten(obss)

    def close_extras(self):
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    @classmethod
    def _flatten(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0

        if isinstance(v[0], dict):
            return {k: np.stack([v_[k] for v_ in v]) for k in v[0].keys()}
        else:
            return np.stack(v)

    @classmethod
    def _flatten_series(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0
        assert all([len(v_) > 0 for v_ in v])

        return [v__ for v_ in v for v__ in v_]


# 支持多智能体版本的 VecEnv，具有共享观察空间的接口。
class ShareVecEnv(VecEnv):
    """
    Multi-agent version of VevEnv, that is, support `share_observation_space` interface.
    """
    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        super().__init__(num_envs, observation_space, action_space)
        self.share_observation_space = share_observation_space


# 多智能体版本的 DummyVecEnv，支持共享观察空间的接口。
class ShareDummyVecEnv(DummyVecEnv, ShareVecEnv):
    """
    Multi-agent version of DummyVecEnv, that is, support `share_observation_space` interface.

    DummyVecEnv is a VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case, avoids communication overhead)
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(self.envs), env.ally_obs_spaces, env.ally_share_obs_spaces, env.ally_action_spaces)
        # ShareVecEnv.__init__(self, len(self.envs), env.ally_observation_spaces, env.ally_share_observation_spaces, env.ally_action_spaces)
        self.actions = None
        self.attention_weight = None
        self.num_agents = getattr(self.envs[0], "num_agents", env.num_allyUAVs)
        # self.num_agents = getattr(self.envs[0], "num_agents", env.ally_num)

    def step_wait(self):
        results = [env.step(a, w) for (a, w, env) in zip(self.actions, self.attention_weight, self.envs)]
        obs, share_obs, ally_attention_n, enemy_attention_n, rews, dones, infos = map(list, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i], share_obs[i], ally_attention_n[i], enemy_attention_n[i] = self.envs[i].reset()
            elif isinstance(done, (list, tuple, np.ndarray)):
                if np.all(done):
                    obs[i], share_obs[i], ally_attention_n[i], enemy_attention_n[i] = self.envs[i].reset()
            elif isinstance(done, dict):
                if np.all(list(done.values())):
                    obs[i], share_obs[i], ally_attention_n[i], enemy_attention_n[i] = self.envs[i].reset()
            else:
                raise NotImplementedError("Unexpected type of done!")
        self.actions = None
        self.attention_weight = None
        return self._flatten(obs), self._flatten(share_obs), self._flatten(ally_attention_n), self._flatten(enemy_attention_n), self._flatten(rews), self._flatten(dones), np.array(infos)

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, ally_attention_n, enemy_attention_n = map(np.array, zip(*results))
        return obs, share_obs, ally_attention_n, enemy_attention_n


def shareworker(remote: Connection, parent_remote: Connection, env_fn_wrappers):
    """Maintain an environment instance in subprocess,
       communicate with parent-process via multiprocessing.Pipe.

    Args:
        remote (Connection): used for current subprocess to send/receive data.
        parent_remote (Connection): used for mainprocess to send/receive data. [Need to be closed in subprocess!]
        env_fn_wrappers (method): functions to create gym.Env instance.
    """
    def step_env(env, action, attention_weight):
        obs, share_obs, ally_attention_n, enemy_attention_n, reward, done, info = env.step(action, attention_weight)
        if 'bool' in done.__class__.__name__:
            if done:
                obs, share_obs, ally_attention_n, enemy_attention_n = env.reset()
        elif isinstance(done, (list, tuple, np.ndarray)):
            if np.all(done):
                obs, share_obs, ally_attention_n, enemy_attention_n = env.reset()
        elif isinstance(done, dict):
            if np.all(list(done.values())):
                obs, share_obs, ally_attention_n, enemy_attention_n = env.reset()
        else:
            raise NotImplementedError("Unexpected type of done!")
        return obs, share_obs, ally_attention_n, enemy_attention_n, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, action, weight = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action, attention_weight) for env, action, attention_weight in zip(envs, action, weight)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper((envs[0].ally_obs_spaces, envs[0].ally_share_obs_spaces, envs[0].ally_action_spaces)))
            elif cmd == 'get_num_agents':
                remote.send(CloudpickleWrapper((getattr(envs[0], "num_agents", envs[0].num_allyUAVs))))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class ShareSubprocVecEnv(SubprocVecEnv, ShareVecEnv):
    def __init__(self, env_fns, context='spawn', in_series=1):
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        # create Pipe connections to send/recv data from subprocesses,
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nremotes)])
        self.ps = [Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None, None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv().x
        ShareVecEnv.__init__(self, nenvs, observation_space, share_observation_space, action_space)

        self.remotes[0].send(('get_num_agents', None, None))
        self.num_agents = self.remotes[0].recv().x

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results)  # [[tuple] * in_series] * nremotes => [tuple] * nenvs
        self.waiting = False
        obs, share_obs, ally_attention_n, enemy_attention_n, rewards, dones, infos = zip(*results)

        return self._flatten(obs), self._flatten(share_obs), self._flatten(ally_attention_n), self._flatten(enemy_attention_n), self._flatten(rewards), self._flatten(dones), np.array(infos)

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None, None))
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results)
        obs, share_obs, ally_attention_n, enemy_attention_n = zip(*results)
        return self._flatten(obs), self._flatten(share_obs), self._flatten(ally_attention_n), self._flatten(enemy_attention_n)
