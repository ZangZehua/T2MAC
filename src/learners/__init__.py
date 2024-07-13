from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .vffac_learner import QLearner as VffacLearner
from .tmac_vffac_learner import QLearner as TMACVffacLearner
from .tmac_full_comm_learner import QLearner as TFCLearner
from .tmac_p2p_comm_learner import QLearner as P2PLearner
from .tmac_comm_rate_learner import QLearner as CRLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["vffac_learner"] = VffacLearner
REGISTRY["tmac_vffac_learner"] = TMACVffacLearner
REGISTRY["tmac_full_comm_learner"] = TFCLearner
REGISTRY['tmac_p2p_comm_learner'] = P2PLearner
REGISTRY['tmac_comm_rate_learner'] = CRLearner