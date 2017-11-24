from sandbox.cpo.gather_env import GatherEnv
from sandbox.cpo.point_env import PointEnv


class PointGatherEnv(GatherEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2
