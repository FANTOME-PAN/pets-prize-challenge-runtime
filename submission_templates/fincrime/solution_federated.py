from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import torch
from flwr.common import FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger
import numpy as np
import pandas as pd
# from torch import nn
from .secagg import public_key_to_bytes, bytes_to_public_key, generate_key_pairs, generate_shared_key, \
    quantize, reverse_quantize, encrypt, decrypt
import pickle
from .fl_logic import TrainClientTemplate

LOGIC_TEST = True
DEBUG = True

"""=== Utility Functions ==="""


def pyobj2bytes(obj: object) -> bytes:
    return pickle.dumps(obj)


def bytes2pyobj(b: bytes) -> object:
    return pickle.loads(b)


def empty_parameters() -> Parameters:
    """Utility function that generates empty Flower Parameters dataclass instance."""
    return ndarrays_to_parameters([])


# train and return the XGBoost on swift_train
def pretrain_swift_model(swift_df: pd.DataFrame):
    raise NotImplementedError()


# return predicted probabilities
def predict_proba(net, x) -> np.ndarray:
    raise NotImplementedError()


def masking(seed_offset, x: np.ndarray, cid, shared_seed_dict: Dict, target_range=(1 << 32)) \
        -> np.ndarray:
    for other_cid, seed in shared_seed_dict.items():
        np.random.seed(seed + seed_offset)
        msk = np.random.randint(0, target_range, x.shape, dtype=np.int64)
        if cid < other_cid:
            x += msk
        else:
            x -= msk
    return x


class DataLoader:
    def __init__(self, df: pd.DataFrame, batch_size=4):
        self.df = df
        self.account2bank = {}
        self.bank2cid = {}
        self.batch_size = batch_size
        self.ptr = 0

    def _build_account2bank_dict(self, df: pd.DataFrame):
        prv_t = ['UETR', 'Sender', 'Receiver', 'OrderingAccount', 'BeneficiaryAccount']
        for t in df[['UETR', 'Sender', 'Receiver', 'OrderingAccount',
                     'BeneficiaryAccount']].values:
            uetr, sender, receiver, oa, ba = t
            if ba in self.account2bank and oa in self.account2bank:
                prv_t = t
                continue
            if uetr == prv_t[0]:
                assert sender == self.account2bank[ba]
                logger.info("Multiple individual transaction detected")
                self.account2bank[ba] = receiver
            else:
                self.account2bank[oa] = sender
                self.account2bank[ba] = receiver
            prv_t = t
        for account in self.account2bank:
            self.account2bank[account] = self.bank2cid[self.account2bank[account]]

    def set_bank2cid(self, bank2cid: Dict[str, str]):
        self.bank2cid = bank2cid
        self._build_account2bank_dict(self.df)

    # return input batch to XGBoost and
    # bathed (sender_client_cid, receiver_client_cid, ordering_account, beneficiary_account)
    def next_batch(self) -> Tuple[np.ndarray, List[Tuple[str, str, str, str]]]:
        data = self.df.values[self.ptr: self.ptr + self.batch_size]
        oa_ba = self.df[['OrderingAccount', 'BeneficiaryAccount']].values[self.ptr: self.ptr + self.batch_size]
        self.ptr += self.batch_size
        if self.ptr >= len(self.df):
            self.ptr = 0
        batch = [(self.account2bank[oa], self.account2bank[ba], oa, ba) for oa, ba in oa_ba]
        return data, batch


def try_decrypt_and_load(key, ciphertext):
    try:
        plaintext = decrypt(key, ciphertext)
        ret = bytes2pyobj(plaintext)
        return ret
    except:
        return None


# class TrainSwiftClient(fl.client.NumPyClient):
#     """Custom Flower NumPyClient class for training."""
#     def __init__(
#         self, cid: str, swift_df: pd.DataFrame, model: SwiftModel, client_dir: Path
#     ):
#         super().__init__()
#         self.cid = cid
#         self.swift_df = swift_df
#         self.model = model
#         self.client_dir = client_dir


"""=== Client Class ==="""


class TrainSwiftClient(TrainClientTemplate):
    """Custom Flower NumPyClient class for training."""

    def __init__(
            self, cid: str, df: pd.DataFrame, model, client_dir: Path
    ):
        super().__init__(cid, df, model, client_dir)
        # UNIMPLEMENTED CODE HERE
        self.weights = np.random.rand(28)
        if LOGIC_TEST:
            self.weights = np.zeros(28)
        self.loader = DataLoader(df)
        self.df = df
        self.net = None
        self.bank2cid = {}
        self.lr = 0.1
        self.agg_grad = np.zeros(28)
        self.proba = np.array(0)
        # with open(self.client_dir / 'a.txt', 'r') as f:
        #     logger.info(f"\nSWIFT: {f.read()} \n")

    def _update_weights(self, grad: np.ndarray):
        self.weights -= self.lr * grad

    def setup_round2(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        logger.info("swift client: build bank to cid dict")
        for lst in parameters:
            cid = lst[0]
            logger.info(f'swift client: reading bank list of {cid}, sized {len(lst) - 1}')
            self.bank2cid |= dict(zip(lst[1:], [cid] * (len(lst) - 1)))
        self.loader.set_bank2cid(self.bank2cid)
        logger.info("swift client: train XGBoost")
        # training code
        # UNIMPLEMENTED CODE HERE

    def stage0(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        # skip the first stage 0
        if server_rnd > 3:
            logger.info('swift client: updating parameters with received gradients...')
            grad = (parameters[0] + self.partial_grad) & 0xffffffff
            if DEBUG:
                logger.info(f"reconstructed grad = {str(grad)}")

            grad = reverse_quantize([grad], 16, 1 << 24)[0]
            grad -= (len(self.shared_seed_dict) - 1) * 16.
            self.agg_grad[1:-1] = grad
            if DEBUG:
                logger.info(f"aggregate grad = {str(self.agg_grad)}")
            self._update_weights(self.agg_grad)

        logger.info('swift client: preparing batch...')
        x, batch = self.loader.next_batch()
        if LOGIC_TEST:
            proba = np.zeros(x.shape[0])
        else:
            proba = predict_proba(self.net, x)
        self.proba = proba
        # wx = w_0 * proba + b, where b = weights[27]
        wx = quantize([self.weights[0] * proba + self.weights[-1]], 16, 1 << 24)[0]
        masked_wx = masking(server_rnd + 1, wx, self.cid, self.shared_seed_dict)
        logger.info(f'client {self.cid}: masking offset = {server_rnd + 1}')
        ret_dict = {}
        for i, (sender_cid, receiver_cid, ordering_account, beneficiary_account) in enumerate(batch):
            cipher_oa = encrypt(self.shared_secret_dict[sender_cid],
                                pyobj2bytes(ordering_account))
            cipher_ba = encrypt(self.shared_secret_dict[receiver_cid],
                                pyobj2bytes(beneficiary_account))
            t = (cipher_oa, cipher_ba)
            ret_dict[str(i)] = pyobj2bytes(t)

        # UNIMPLEMENTED CODE HERE
        labels = x[:, -1].astype(int)
        return [masked_wx, self.weights[1:-1], labels], 0, ret_dict

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        self.beta = parameters[0]
        self.partial_grad = masking(server_rnd, np.zeros(26, dtype=int), self.cid, self.shared_seed_dict)
        self.agg_grad[-1] = self.beta.sum()
        self.agg_grad[0] = (self.beta * self.proba).sum()
        return [], 0, {}


class TrainBankClient(TrainClientTemplate):
    """Custom Flower NumPyClient class for training."""

    def __init__(
            self, cid: str, df: pd.DataFrame, client_dir: Path
    ):
        super().__init__(cid, df, None, client_dir)
        self.weights = np.array(0)
        self.cached_flags = np.array(0)
        pth = client_dir / 'account2flag.pkl'
        if pth.exists():
            self.account2flag = torch.load(pth)
        else:
            self.account2flag = dict(zip(df['Account'], df['Flags'].astype(int)))
            torch.save(self.account2flag, pth)
        self.bank_lst = list(df['Bank'].unique())

    def _get_flag(self, account: str):
        return self.account2flag.setdefault(account, 12)

    def setup_round1(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        t[0].append(np.array([self.cid] + self.bank_lst, dtype=str))
        logger.info(f'client {self.cid}: upload bank list of size {len(t[0][0]) - 1}')

    def stage1(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        logger.info(f"Client {self.cid}: reading encrypted batch and computing masked results...")
        # logger.info(f"client {self.cid}: received {str(config)}")
        # the dim of weights is 26 at bank client. the weight of proba and the bias are kept in swift client
        # weights[0] to [12] correspond to weights for ordering account flags
        # weights[13] to [25] correspond to weights for beneficiary account flags
        self.weights = parameters[0]
        # try read batch
        ret = np.zeros(len(config))
        key = self.shared_secret_dict['swift']
        # for ret[i] = optional(w2 * x2) + optional(w3 * x3)
        # w2 = weights[:13], w3 = weights[13:]
        # x2, x3 is the one-hot encoding of OA flag, BA flag
        self.cached_flags = np.zeros((len(config), 26))
        for str_i, obj_bytes in config.items():
            # logger.info(f'try decrypting {obj_bytes} of type {type(obj_bytes)}')
            cipher_oa, cipher_ba = bytes2pyobj(obj_bytes)
            i = int(str_i)
            if (oa := try_decrypt_and_load(key, cipher_oa)) is not None:
                flg = self._get_flag(oa)
                ret[i] += self.weights[flg]
                self.cached_flags[i][flg] = 1
                logger.info(f'BATCH_IDX: {i} ORDERING_ACCOUNT')
            if (ba := try_decrypt_and_load(key, cipher_ba)) is not None:
                flg = self._get_flag(ba)
                ret[i] += self.weights[flg + 13]
                self.cached_flags[i][flg + 13] = 1
                logger.info(f'BATCH_IDX: {i} BENEFICIARY_ACCOUNT')
        ret = quantize([ret], clipping_range=16, target_range=(1 << 24))[0]
        masked_ret = masking(server_rnd, ret, self.cid, self.shared_seed_dict)
        logger.info(f'client {self.cid}: masking offset = {server_rnd}')
        return [masked_ret], 0, {}

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        self.beta = parameters[0]
        # bank client code
        # reshape beta to B x 1 from B, then expand it to B x 26
        # do element-wise production between expanded beta and cached_flags (dim: B x 26)
        # finally, sum up along axis 0, get results of dim 26, i.e., the partial agg_grad on this client
        logger.info(f'{self.cached_flags}')
        self.partial_grad = (self.beta.reshape((-1, 1)).repeat(26, axis=1) * self.cached_flags).sum(axis=0)
        self.partial_grad = quantize([self.partial_grad], 16, 1 << 24)[0]
        masked_partial_grad = masking(server_rnd, self.partial_grad, self.cid, self.shared_seed_dict)
        logger.info(f'client {self.cid}: uploading masked grad : {masked_partial_grad}')
        return [masked_partial_grad], 0, {}


def train_client_factory(
        cid: str,
        data_path: Path,
        client_dir: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for training.
    The federated learning simulation engine will use this function to
    instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        data_path (Path): Path to CSV data file specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    if cid == "swift":
        logger.info("Initializing SWIFT client for {}", cid)
        swift_df = pd.read_csv(data_path, index_col="MessageId")
        # UNIMPLEMENTED CODE HERE
        model = None
        return TrainSwiftClient(
            cid, df=swift_df, model=model, client_dir=client_dir
        )
    else:
        logger.info("Initializing bank client for {}", cid)
        bank_df = pd.read_csv(data_path, dtype=pd.StringDtype())
        return TrainBankClient(
            cid, df=bank_df, client_dir=client_dir
        )


"""=== Strategy Class ==="""


class TrainStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for training."""

    def __init__(self, server_dir: Path, num_rounds):
        self.server_dir = server_dir
        self.public_keys_dict = {}
        self.fwd_dict = {}
        self.agg_grad = np.zeros(26)
        self.stage = 0
        self.num_rnds = num_rounds
        self.label = np.array(0)
        self.weights = np.array(0)
        self.logit = np.array(0)
        self.beta = np.array(0)
        self.encrypted_batch: Dict[str, bytes] = {}
        self.cached_banklsts = []
        super().__init__()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return empty_parameters()

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        cid_dict: Dict[str, ClientProxy] = client_manager.all()
        config_dict = {"round": server_round}
        logger.info(f"[START] server round {server_round}")
        logger.info(f'GPU STATUS: {torch.cuda.is_available()}')
        # rnd 1
        # collect public keys
        if server_round == 1:
            # add all cids to the config_dict as keys
            config_dict = dict(zip(cid_dict.keys(), [0] * len(cid_dict))) | config_dict
            logger.info(f"server's requesting public keys...")
            if DEBUG:
                logger.info(f"send to clients {str(config_dict)}")
            fit_ins = FitIns(parameters=empty_parameters(), config=config_dict)
            return [(o, fit_ins) for o in cid_dict.values()]
        # rnd 2
        # broadcast public keys, swift train
        if server_round == 2:
            # forward public keys to corresponding clients
            logger.info(f"server's forwarding public keys...")
            for cid in self.fwd_dict:
                logger.info(f'forward to {cid} {str(self.fwd_dict[cid].keys())}')
            ins_lst = [(
                proxy,
                FitIns(parameters=empty_parameters() if proxy.cid != 'swift' else self.cached_banklsts
                       , config=self.fwd_dict[cid] | config_dict)
            ) for cid, proxy in cid_dict.items()]
            if DEBUG:
                logger.info(f"server's sending to swift bank lists {str(parameters_to_ndarrays(self.cached_banklsts))}")
            self.cached_banklsts = []
            self.fwd_dict = {}
            return ins_lst
        # rnd 3 -> N
        # joint train
        config_dict['stage'] = self.stage
        if self.stage == 0:
            logger.info(f"stage 0: sending gradients to swift client")
            if server_round == self.num_rnds:
                config_dict['stop'] = 1
            fit_ins = FitIns(parameters=ndarrays_to_parameters([self.agg_grad]), config=config_dict)
            ins_lst = [(cid_dict['swift'], fit_ins)]
        elif self.stage == 1:
            logger.info(f"stage 1: broadcasting model weights and encrypted batch to bank clients")
            # broadcast the weights and the encrypted batch to all bank clients
            config_dict = config_dict | self.encrypted_batch
            fit_ins = FitIns(parameters=ndarrays_to_parameters([self.weights]), config=config_dict)
            ins_lst = [(proxy, fit_ins) for cid, proxy in cid_dict.items() if cid != 'swift']
        elif self.stage == 2:
            logger.info(f"stage 2: broadcasting beta to all clients {self.beta}")
            fit_ins = FitIns(parameters=ndarrays_to_parameters([self.beta]), config=config_dict)
            ins_lst = [(proxy, fit_ins) for cid, proxy in cid_dict.items()]
        else:
            raise AssertionError("Stage number should be 0, 1, or 2")

        return ins_lst

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        if (n_failures := len(failures)) > 0:
            logger.error(f"Had {n_failures} failures in round {server_round}")
            raise Exception(f"Had {n_failures} failures in round {server_round}")
        # rnd 1
        # gather all public keys, and forward them to corresponding client
        if server_round == 1:
            logger.info(f'server\'s processing received bank lists')
            # bank client would store the list of bank names in the parameters field
            self.cached_banklsts = [parameters_to_ndarrays(res.parameters)[0]
                                    for proxy, res in results if proxy.cid != 'swift']
            self.cached_banklsts = ndarrays_to_parameters(self.cached_banklsts)
            # fwd_dict[to_client][from_client] = public key to to_client generated by from_client
            logger.info(f'server\'s creating forward dict')
            self.fwd_dict = dict([(o.cid, {}) for o, _ in results])
            for client, res in results:
                for other_cid, pk_bytes in res.metrics.items():
                    self.fwd_dict[other_cid][client.cid] = pk_bytes
            logger.info(f"[END] server round {server_round}")
            return None, {}
        # rnd 2
        # do nothing
        if server_round == 2:
            logger.info(f"[END] server round {server_round}")
            return None, {}
        # rnd 3 -> N, joint training
        if self.stage == 0:
            if server_round < self.num_rnds:
                masked_wx, weights, label = parameters_to_ndarrays(results[0][1].parameters)
                self.logit = masked_wx
                self.label = label
                self.weights = weights
                # broadcast to all bank clients
                self.encrypted_batch = results[0][1].metrics
                logger.info(f"server received encrypted batch:\n {self.encrypted_batch}")
            # if server_round == self.num_rnds, do nothing
            pass
        elif self.stage == 1:
            # receive masked results from all bank clients
            for proxy, res in results:
                masked_res = parameters_to_ndarrays(res.parameters)[0]
                self.logit += masked_res
            self.logit &= 0xffffffff
            self.logit = reverse_quantize([self.logit], clipping_range=16, target_range=(1 << 24))[0]
            self.logit -= len(results) * 16.
            logger.info(f'server: reconstructed logits = {self.logit}')
            logger.info(f'server: labels = {self.label}')

            tmp = np.exp(-self.logit)
            y_pred = 1. / (1. + tmp)
            beta = np.zeros(len(self.encrypted_batch))
            msk = (self.label == 1)
            beta[msk] = 1. + tmp[msk]
            beta[~msk] = 1. / (y_pred[~msk] - 1.)
            beta *= (tmp * y_pred * y_pred)
            self.beta = beta
            pass
        elif self.stage == 2:
            self.agg_grad = np.zeros(26, dtype=int)
            for client, res in results:
                if client.cid == 'swift':
                    continue
                t = parameters_to_ndarrays(res.parameters)
                logger.info(f'server received {t}')
                self.agg_grad += t[0]
            pass
        else:
            raise AssertionError("Stage number should be 0, 1, or 2")
        self.stage = (self.stage + 1) % 3
        return None, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Not running any federated evaluation."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        """Not aggregating any evaluation."""
        return None

    def evaluate(self, server_round, parameters):
        """Not running any centralized evaluation."""
        return None


def train_strategy_factory(
        server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federated learning rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    # num_rounds of setup phase = 2
    # num_rounds of one training round = 3 (stage 0,1,2)
    # end round, end at stage 0
    # num_rounds = 2 + 3N + 1 = 3 + 3N
    num_rounds = 12
    training_strategy = TrainStrategy(server_dir=server_dir, num_rounds=num_rounds)
    return training_strategy, num_rounds


class TestClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for test."""


def test_client_factory(
        cid: str,
        data_path: Path,
        client_dir: Path,
        preds_format_path: Path,
        preds_dest_path: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for test-time
    inference. The federated learning simulation engine will use this function
    to instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages. The
            SWIFT node will always be named 'swift'.
        data_path (Path): Path to CSV test data file specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.
        preds_format_path (Optional[Path]): Path to CSV file matching the format
            you must write your predictions with, filled with dummy values. This
            will only be non-None for the 'swift' client—bank clients should not
            write any predictions and receive None for this argument.
        preds_dest_path (Optional[Path]): Destination path that you must write
            your test predictions to as a CSV file. This will only be non-None
            for the 'swift' client—bank clients should not write any predictions
            and will receive None for this argument.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    ...
    return TestClient(...)


class TestStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for test."""


def test_strategy_factory(
        server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federation rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    ...
    return TestStrategy(...), 1
