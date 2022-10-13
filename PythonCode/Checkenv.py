from tabnanny import check
from stable_baselines3.common.env_checker import check_env
from FakeNewsEnv import FakeNewsEnv

env = FakeNewsEnv()

check_env(env)