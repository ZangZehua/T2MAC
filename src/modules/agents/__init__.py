REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_msg_agent import RnnMsgAgent
from .tmac_rnn_agent import RNNAgent as TMACRNNAgent
from .tmac_rnn_msg_agent import RnnMsgAgent as TMACMsgAgent
from .tmac_full_comm_rnn_msg_agent import RnnMsgAgent as TFCMsgAgent
from .tmac_p2p_comm_rnn_msg_agent import RnnMsgAgent as P2PMsgAgent
from .tmac_comm_rate_rnn_msg_agent import RnnMsgAgent as CRMsgAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY['rnn_msg'] = RnnMsgAgent
REGISTRY['tmac_rnn'] = TMACRNNAgent
REGISTRY['tmac_vffac_rnn_msg'] = RnnMsgAgent
REGISTRY['tmac_full_comm_rnn_msg'] = TFCMsgAgent
REGISTRY['tmac_p2p_comm_rnn_msg'] = P2PMsgAgent
REGISTRY['tmac_comm_rate_rnn_msg'] = CRMsgAgent