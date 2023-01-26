from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set

import flwr as fl
import torch
from flwr.common import FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, \
    EvaluateIns, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger
import numpy as np
import pandas as pd
from torch import nn
from .secagg import quantize, reverse_quantize, encrypt, decrypt
import pickle
from .fl_logic import TrainClientTemplate
from .fl_xgboost_utils import fit_swift, test_swift
from .fl_utils import pyobj2bytes, bytes2pyobj, bytes_to_ndarrays
from .settings import DEBUG, LOGIC_TEST
import time


class Timer:
    def __init__(self):
        self.mark = 0
        self.toc_cnt = 0

    def tic(self):
        self.mark = time.time()

    def toc(self):
        return time.time() - self.mark


timer = Timer()

# num_rounds = 1 + 1.5N + 0.5 = 1.5(N+1)
REAL_ROUNDS = 201  # odd
TRAINING_ROUNDS = int((REAL_ROUNDS + 1) * 3 // 2)
LEARNING_RATE = 1.0
BATCH_SIZE = 2048
CLIP_RANGE = 64
TARGET_RANGE = 1 << 27

"""=== Utility Functions ==="""


def empty_parameters() -> Parameters:
    """Utility function that generates empty Flower Parameters dataclass instance."""
    return ndarrays_to_parameters([])


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


class TrainDataLoader:
    def __init__(self, batch_size=BATCH_SIZE):
        self.df: pd.DataFrame = None
        self.proba = None
        self.account2cid = {}
        self.bank2cid: List[Tuple[set, str]] = []
        self.batch_size = batch_size
        self.ptr = 0

    def _build_account2bank_dict(self, df: pd.DataFrame):
        bank2cid = self.bank2cid
        # logger.info(f'Bank 2 cid:\n{str(bank2cid)}')
        for bankset, cid in bank2cid:
            logger.info(f'{cid}, set of size {len(bankset)}')

        def pick(bank):
            bank = str(bank)
            for bankset, cid in bank2cid:
                if bank in bankset:
                    return cid
            logger.info(f'{bank}: Bank name not found')
            return 'missing'

        for t in df[['Sender', 'Receiver', 'OrderingAccount',
                     'BeneficiaryAccount']].values:
            sender, receiver, oa, ba = t
            self.account2cid[oa] = pick(sender)
            self.account2cid[ba] = pick(receiver)

    def set(self, df: pd.DataFrame, bank2cid: List[Tuple[set, str]]):
        self.bank2cid = bank2cid
        self._build_account2bank_dict(df)
        df = df[['OrderingAccount', 'BeneficiaryAccount', 'Label']]
        df = df.reset_index()
        self.df = df[['OrderingAccount', 'BeneficiaryAccount', 'Label']]

    def shuffle(self):
        self.ptr = 0
        idx = np.random.permutation(self.df.index)
        self.proba = self.proba[idx]
        self.df = self.df.iloc[idx]

    def set_proba(self, proba: np.ndarray):
        assert proba.shape[0] == len(self.df)
        self.proba = proba

    # return predicted probabilities of the batch and
    # bathed (sender_client_cid, receiver_client_cid, ordering_account, beneficiary_account) and labels
    def next_batch(self) -> Tuple[np.ndarray, List[Tuple[str, str, str, str]], np.ndarray]:
        batch_proba = self.proba[self.ptr: self.ptr + self.batch_size]
        oa_ba_label = self.df.values[self.ptr: self.ptr + self.batch_size]
        self.ptr += self.batch_size
        if self.ptr >= len(self.df):
            self.shuffle()
        batch = [(self.account2cid[oa], self.account2cid[ba], oa, ba) for oa, ba, _ in oa_ba_label]
        return batch_proba, batch, oa_ba_label[:, -1].astype(int)


class TestDataLoader:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.proba = None
        self.account2cid = {}
        self.bank2cid: List[Tuple[set, str]] = []
        self.ptr = 0

    def _build_account2bank_dict(self, df: pd.DataFrame):
        bank2cid = self.bank2cid

        def pick(bank):
            bank = str(bank)
            for bankset, cid in bank2cid:
                if bank in bankset:
                    return cid
            logger.info(f'{bank}: Bank name not found')
            return 'missing'

        for t in df[['Sender', 'Receiver', 'OrderingAccount',
                     'BeneficiaryAccount']].values:
            sender, receiver, oa, ba = t
            self.account2cid[oa] = pick(sender)
            self.account2cid[ba] = pick(receiver)

    def set(self, df: pd.DataFrame, bank2cid: List[Tuple[set, str]]):
        self.bank2cid = bank2cid
        self._build_account2bank_dict(df)
        df = df[['OrderingAccount', 'BeneficiaryAccount']]
        df = df.reset_index()
        self.df = df[['OrderingAccount', 'BeneficiaryAccount']]

    def set_proba(self, proba: np.ndarray):
        assert proba.shape[0] == len(self.df)
        self.proba = proba

    # return predicted probabilities and
    # list of (sender_client_cid, receiver_client_cid, ordering_account, beneficiary_account)
    def get_data(self) -> Tuple[np.ndarray, List[Tuple[str, str, str, str]]]:
        if LOGIC_TEST:
            self.proba = np.zeros(len(self.df))
        oa_ba = self.df.values
        batch = [(self.account2cid[oa], self.account2cid[ba], oa, ba) for oa, ba in oa_ba]
        return self.proba, batch


def try_decrypt_and_load(key, ciphertext):
    try:
        plaintext = decrypt(key, ciphertext)
        ret = bytes2pyobj(plaintext)
        return ret
    except:
        return None


"""=== Client Class ==="""


class TrainSwiftClient(TrainClientTemplate):
    """Custom Flower NumPyClient class for training."""

    def __init__(
            self, cid: str, df_pth: Path, model, client_dir: Path
    ):
        # df = df.sample(frac=1)
        super().__init__(cid, df_pth, model, client_dir)
        # UNIMPLEMENTED CODE HERE
        self.weights = np.random.rand(28)
        if LOGIC_TEST:
            self.weights = np.zeros(28)
        self.loader = TrainDataLoader()
        self.bank2cid = []
        self.lr = LEARNING_RATE
        self.agg_grad = np.zeros(28)
        self.proba = np.array(0)

    def _update_weights(self, grad: np.ndarray):
        self.weights -= self.lr * grad

    def setup_round2(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        logger.info("swift client: build bank to cid dict")
        # for lst in parameters:
        #     cid = lst[0]
        #     logger.info(f'swift client: reading bank list of {cid}, sized {len(lst) - 1}')
        #     self.bank2cid |= dict(zip(lst[1:], [cid] * (len(lst) - 1)))
        self.bank2cid = [(set(lst[1:]), lst[0]) for lst in parameters]
        df = pd.read_csv(self.df_pth, index_col='MessageId')
        self.loader.set(df, self.bank2cid)
        logger.info("swift client: train XGBoost")
        # training code
        # UNIMPLEMENTED CODE HERE
        if not LOGIC_TEST:
            # self.loader.set_proba(np.zeros(len(df)))
            all_proba = fit_swift(df, self.client_dir)[1]
            self.loader.set_proba(all_proba)
        else:
            self.loader.set_proba(np.zeros(len(df)))
        self.loader.shuffle()

    def stage0(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        # skip the first stage 0
        if server_rnd > 3:
            logger.info('swift client: updating parameters with received gradients...')
            grad = (parameters[0] + self.partial_grad) & 0xffffffff
            if DEBUG:
                logger.info(f"reconstructed grad = {str(grad)}")

            grad = reverse_quantize([grad], CLIP_RANGE, TARGET_RANGE)[0]
            grad -= (len(self.shared_seed_dict) - 1) * CLIP_RANGE
            self.agg_grad[1:-1] = grad
            # reduction = mean
            self.agg_grad /= self.loader.batch_size
            self.agg_grad = -self.agg_grad
            if DEBUG:
                logger.info(f"aggregate grad = {str(self.agg_grad)}")
            self._update_weights(self.agg_grad)

        logger.info('swift client: preparing batch and masked vectors...')
        proba, batch, labels = self.loader.next_batch()
        # if LOGIC_TEST:
        #     proba = np.zeros(x.shape[0])
        # else:
        #     proba = predict_proba(self.net, x)
        self.proba = proba
        if DEBUG:
            logger.info(f'swift predicted proba: {proba}')
        # wx = w_0 * proba + b, where b = weights[27]
        wx = quantize([self.weights[0] * proba + self.weights[-1]], CLIP_RANGE, TARGET_RANGE)[0]
        masked_wx = masking(server_rnd + 1, wx, self.cid, self.shared_seed_dict)
        if DEBUG:
            logger.info(f'client {self.cid}: masking offset = {server_rnd + 1}')
        ret_dict = {}
        if DEBUG:
            timer.tic()
        first_cid = None
        for first_cid in self.shared_secret_dict:
            break
        for i, (sender_cid, receiver_cid, ordering_account, beneficiary_account) in enumerate(batch):
            if sender_cid == 'missing':
                if DEBUG:
                    logger.info(f'swift client: OrderingAccount {ordering_account} belongs to a missing bank.'
                                f' send to client {first_cid} instead')
                oa_seed = self.shared_secret_dict[first_cid]
            else:
                oa_seed = self.shared_secret_dict[sender_cid]

            if receiver_cid == 'missing':
                if DEBUG:
                    logger.info(f'swift client: BeneficiaryAccount {beneficiary_account} belongs to a missing bank.'
                                f' send to client {first_cid} instead')
                ba_seed = self.shared_secret_dict[first_cid]
            else:
                ba_seed = self.shared_secret_dict[receiver_cid]

            cipher_oa = encrypt(oa_seed,
                                pyobj2bytes(ordering_account))
            cipher_ba = encrypt(ba_seed,
                                pyobj2bytes(beneficiary_account))
            t = (cipher_oa, cipher_ba)
            ret_dict[str(i)] = pyobj2bytes(t)
        if DEBUG:
            logger.info(f'Encryption time: {timer.toc()}')
        # UNIMPLEMENTED CODE HERE
        # labels = x[:, -1].astype(int)
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
            self, cid: str, df_pth: Path, client_dir: Path
    ):
        super().__init__(cid, df_pth, None, client_dir)
        self.weights = np.array(0)
        self.cached_flags = np.array(0)
        self.account2flag = None

    def _get_flag(self, account: str):
        return self.account2flag.setdefault(account, 12)

    def setup_round1(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        df = pd.read_csv(self.df_pth, dtype=pd.StringDtype())
        self.account2flag = dict(zip(df['Account'], df['Flags'].astype(int)))
        t[0].append(np.array([self.cid] + list(df['Bank'].unique()), dtype=str))
        if DEBUG:
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
        if DEBUG:
            timer.tic()
        for str_i, obj_bytes in config.items():
            # logger.info(f'try decrypting {obj_bytes} of type {type(obj_bytes)}')
            cipher_oa, cipher_ba = bytes2pyobj(obj_bytes)
            i = int(str_i)
            if (oa := try_decrypt_and_load(key, cipher_oa)) is not None:
                flg = self._get_flag(oa)
                ret[i] += self.weights[flg]
                self.cached_flags[i][flg] = 1
                if DEBUG and LOGIC_TEST:
                    logger.info(f'BATCH_IDX: {i} ORDERING_ACCOUNT')
            if (ba := try_decrypt_and_load(key, cipher_ba)) is not None:
                flg = self._get_flag(ba)
                ret[i] += self.weights[flg + 13]
                self.cached_flags[i][flg + 13] = 1
                if DEBUG and LOGIC_TEST:
                    logger.info(f'BATCH_IDX: {i} BENEFICIARY_ACCOUNT')
        if DEBUG:
            logger.info(f'Decryption time: {timer.toc()}')
        ret = quantize([ret], CLIP_RANGE, TARGET_RANGE)[0]
        masked_ret = masking(server_rnd, ret, self.cid, self.shared_seed_dict)
        if DEBUG:
            logger.info(f'client {self.cid}: masking offset = {server_rnd}')
        return [masked_ret], 0, {}

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        self.beta = parameters[0]
        # bank client code
        # reshape beta to B x 1 from B, then expand it to B x 26
        # do element-wise production between expanded beta and cached_flags (dim: B x 26)
        # finally, sum up along axis 0, get results of dim 26, i.e., the partial agg_grad on this client
        if DEBUG and LOGIC_TEST:
            logger.info(f'{self.cached_flags}')
        self.partial_grad = (self.beta.reshape((-1, 1)).repeat(26, axis=1) * self.cached_flags).sum(axis=0)
        self.partial_grad = quantize([self.partial_grad], CLIP_RANGE, TARGET_RANGE)[0]
        masked_partial_grad = masking(server_rnd, self.partial_grad, self.cid, self.shared_seed_dict)
        if DEBUG:
            logger.info(f'client {self.cid}: uploading masked grad : {masked_partial_grad}')
        else:
            logger.info(f'client {self.cid}: uploading masked gradient')
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
        return TrainSwiftClient(
            cid, df_pth=data_path, model=None, client_dir=client_dir
        )
    else:
        logger.info("Initializing bank client for {}", cid)
        return TrainBankClient(
            cid, df_pth=data_path, client_dir=client_dir
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
        self.num_rnds = TRAINING_ROUNDS << 1
        self.label = np.array(0)
        self.weights = np.array(0)
        self.logit = np.array(0)
        self.beta = np.array(0)
        self.encrypted_batch: Dict[str, bytes] = {}
        self.cached_banklsts = []
        self.rnd_cnt = 0
        super().__init__()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return empty_parameters()

    def __configure_round(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, Union[FitIns]]]:
        """Configure the next round of training."""
        cid_dict: Dict[str, ClientProxy] = client_manager.all()
        config_dict = {"round": server_round}
        logger.info(f"[START] server round {server_round}")
        if DEBUG:
            logger.info(f'GPU STATUS: {torch.cuda.is_available()}')
        # rnd 1
        # collect public keys
        if server_round == 1:
            # add all cids to the config_dict as keys
            config_dict = dict(zip(cid_dict.keys(), [0] * len(cid_dict))) | config_dict
            logger.info(f"server's requesting public keys...")
            if DEBUG and LOGIC_TEST:
                logger.info(f"send to clients {str(config_dict)}")
            fit_ins = FitIns(parameters=empty_parameters(), config=config_dict)
            return [(o, fit_ins) for o in cid_dict.values()]
        # rnd 2
        # broadcast public keys, swift train
        if server_round == 2:
            # forward public keys to corresponding clients
            logger.info(f"server's forwarding public keys...")

            if DEBUG:
                for cid in self.fwd_dict:
                    logger.info(f'forward to {cid} {str(self.fwd_dict[cid].keys())}')
            ins_lst = [(
                proxy,
                FitIns(parameters=empty_parameters() if proxy.cid != 'swift' else self.cached_banklsts
                       , config=self.fwd_dict[cid] | config_dict)
            ) for cid, proxy in cid_dict.items()]
            if DEBUG and LOGIC_TEST:
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
            if DEBUG:
                logger.info(f"stage 2: broadcasting beta to all clients {self.beta}")
            else:
                logger.info(f"stage 2: broadcasting beta to all clients")
            fit_ins = FitIns(parameters=ndarrays_to_parameters([self.beta]), config=config_dict)
            ins_lst = [(proxy, fit_ins) for cid, proxy in cid_dict.items()]
        else:
            raise AssertionError("Stage number should be 0, 1, or 2")

        return ins_lst

    def __aggregate_round(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
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
                logger.info(f"logit type {type(self.logit)}, dtype {self.logit.dtype}")
                self.label = label
                self.weights = weights
                # broadcast to all bank clients
                self.encrypted_batch = results[0][1].metrics
                if DEBUG and LOGIC_TEST:
                    logger.info(f"server received encrypted batch:\n {self.encrypted_batch}")
            # if server_round == self.num_rnds, do nothing
            pass
        elif self.stage == 1:
            # receive masked results from all bank clients
            for proxy, res in results:
                masked_res = parameters_to_ndarrays(res.parameters)[0]
                self.logit += masked_res
                logger.info(f"logit type {type(self.logit)}, dtype {self.logit.dtype}")
                logger.info(f"masked_res type {type(masked_res)}, dtype {masked_res.dtype}")
            self.logit &= 0xffffffff
            self.logit = reverse_quantize([self.logit], CLIP_RANGE, TARGET_RANGE)[0]
            self.logit -= len(results) * CLIP_RANGE
            if DEBUG:
                logger.info(f'server: reconstructed logits = {self.logit}')
                logger.info(f'server: labels = {self.label}')

            y_pred = np.zeros(len(self.logit), dtype=np.float64)
            msk = self.logit >= 0
            y_pred[msk] = 1. / (1. + np.exp(-self.logit[msk]))
            e_logit = np.exp(self.logit[~msk])
            y_pred[~msk] = e_logit / (1. + e_logit)
            # y_pred = 1. / (1. + tmp)
            if DEBUG:
                logger.info(f'server: preds = {y_pred}')
            # loss = nn.BCELoss()(y_pred, self.label)
            y = self.label
            loss = -((1 - y) * np.log(1. - y_pred) + y * np.log(y_pred)).mean()
            logger.info(f'BCE loss: {loss}')
            beta = np.zeros(len(self.encrypted_batch))
            exp_minus_logit = np.exp(-self.logit)
            msk = (self.label == 1)
            beta[msk] = 1. + exp_minus_logit[msk]
            beta[~msk] = 1. / (y_pred[~msk] - 1.)
            beta *= (exp_minus_logit * y_pred * y_pred)
            self.beta = beta
            pass
        elif self.stage == 2:
            self.agg_grad = np.zeros(26, dtype=int)
            for client, res in results:
                if client.cid == 'swift':
                    continue
                t = parameters_to_ndarrays(res.parameters)
                if DEBUG:
                    logger.info(f'server received {t}')
                self.agg_grad += t[0]
            pass
        else:
            raise AssertionError("Stage number should be 0, 1, or 2")
        self.stage = (self.stage + 1) % 3
        return None, {}

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.rnd_cnt += 1
        return self.__configure_round(self.rnd_cnt, parameters, client_manager)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        return self.__aggregate_round(self.rnd_cnt, results, failures)

    def configure_evaluate(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        self.rnd_cnt += 1
        ins_lst = self.__configure_round(self.rnd_cnt, parameters, client_manager)
        return [(proxy, EvaluateIns(ins.parameters, ins.config)) for proxy, ins in ins_lst]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        def convert(res: EvaluateRes) -> FitRes:
            params = bytes2pyobj(res.metrics.pop('parameters'))
            return FitRes(res.status, params, res.num_examples, res.metrics)

        results = [(proxy, convert(eval_res)) for proxy, eval_res in results]
        return self.__aggregate_round(self.rnd_cnt, results, failures)

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
    # num_rounds of setup phase = 2 / 2 = 1
    # num_rounds of one training round = 3 / 2 = 1.5 (stage 0,1,2)
    # end round, end at stage 0, i.e., 0.5 round
    # num_rounds = 1 + 1.5N + 0.5 = 1.5(N+1)
    num_rounds = TRAINING_ROUNDS
    training_strategy = TrainStrategy(server_dir=server_dir, num_rounds=num_rounds)
    return training_strategy, num_rounds


class TestSwiftClient(TrainClientTemplate):
    """Custom Flower NumPyClient class for test."""

    def __init__(
            self, cid: str, df_pth: Path, client_dir: Path,
            preds_format_path: Path,
            preds_dest_path: Path,
    ):
        super().__init__(cid, df_pth, None, client_dir)
        # UNIMPLEMENTED CODE HERE
        self.weights = np.random.rand(28)
        if LOGIC_TEST:
            self.weights = np.zeros(28)
        self.loader = TestDataLoader()
        self.bank2cid: List[Tuple[Set[str], str]] = []
        self.proba = np.array(0)
        self.preds_format_path = preds_format_path
        self.preds_dest_path = preds_dest_path
        self.index = None

    def setup_round2(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        logger.info("swift client: build bank to cid dict")
        self.bank2cid = [(set(lst[1:]), lst[0]) for lst in parameters]
        df = pd.read_csv(self.df_pth, index_col='MessageId')
        self.index = df.index
        self.loader.set(df, self.bank2cid)
        logger.info("swift client: test XGBoost")
        # training code
        # UNIMPLEMENTED CODE HERE
        if not LOGIC_TEST:
            # self.loader.set_proba(np.zeros(len(df)))
            all_proba = test_swift(df, self.client_dir)[1]
            self.loader.set_proba(all_proba)
        else:
            self.loader.set_proba(np.zeros(len(df)))

    def stage0(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        logger.info('swift client: preparing data...')
        proba, batch = self.loader.get_data()
        # if LOGIC_TEST:
        #     proba = np.zeros(x.shape[0])
        # else:
        #     proba = predict_proba(self.net, x)
        self.proba = proba
        if DEBUG:
            logger.info(f'swift predicted proba: {proba}')
        # wx = w_0 * proba + b, where b = weights[27]
        wx = quantize([self.weights[0] * proba + self.weights[-1]], CLIP_RANGE, TARGET_RANGE)[0]
        masked_wx = masking(server_rnd + 1, wx, self.cid, self.shared_seed_dict)
        if DEBUG:
            logger.info(f'client {self.cid}: masking offset = {server_rnd + 1}')
        ret_dict = {}
        first_cid = None
        for first_cid in self.shared_secret_dict:
            break
        for i, (sender_cid, receiver_cid, ordering_account, beneficiary_account) in enumerate(batch):
            if sender_cid == 'missing':
                if DEBUG:
                    logger.info(f'swift client: OrderingAccount {ordering_account} belongs to a missing bank.'
                                f' send to client {first_cid} instead')
                oa_seed = self.shared_secret_dict[first_cid]
            else:
                oa_seed = self.shared_secret_dict[sender_cid]

            if receiver_cid == 'missing':
                if DEBUG:
                    logger.info(f'swift client: BeneficiaryAccount {beneficiary_account} belongs to a missing bank.'
                                f' send to client {first_cid} instead')
                ba_seed = self.shared_secret_dict[first_cid]
            else:
                ba_seed = self.shared_secret_dict[receiver_cid]

            cipher_oa = encrypt(oa_seed,
                                pyobj2bytes(ordering_account))
            cipher_ba = encrypt(ba_seed,
                                pyobj2bytes(beneficiary_account))
            t = (cipher_oa, cipher_ba)
            ret_dict[str(i)] = pyobj2bytes(t)

        # UNIMPLEMENTED CODE HERE
        # labels = x[:, -1].astype(int)
        return [masked_wx, self.weights[1:-1]], 0, ret_dict

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        y_pred = parameters[0]
        final_preds = pd.Series(y_pred, index=self.index)
        preds_format_df = pd.read_csv(self.preds_format_path, index_col="MessageId")
        preds_format_df["Score"] = preds_format_df.index.map(final_preds)
        preds_format_df["Score"] = preds_format_df["Score"].astype(np.float64)
        logger.info("Writing out test predictions...")
        preds_format_df.to_csv(self.preds_dest_path)
        logger.info("Done.")
        return [], 0, {}


class TestBankClient(TrainClientTemplate):
    """Custom Flower NumPyClient class for training."""

    def __init__(
            self, cid: str, df_pth: Path, client_dir: Path
    ):
        super().__init__(cid, df_pth, None, client_dir)
        self.weights = np.array(0)
        self.account2flag = None

    def _get_flag(self, account: str):
        return self.account2flag.setdefault(account, 12)

    def setup_round1(self, parameters, config, t: Tuple[List[np.ndarray], int, dict]):
        df = pd.read_csv(self.df_pth, dtype=pd.StringDtype())
        self.account2flag = dict(zip(df['Account'], df['Flags'].astype(int)))
        t[0].append(np.array([self.cid] + list(df['Bank'].unique()), dtype=str))
        if DEBUG:
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
        for str_i, obj_bytes in config.items():
            # logger.info(f'try decrypting {obj_bytes} of type {type(obj_bytes)}')
            cipher_oa, cipher_ba = bytes2pyobj(obj_bytes)
            i = int(str_i)
            if (oa := try_decrypt_and_load(key, cipher_oa)) is not None:
                flg = self._get_flag(oa)
                ret[i] += self.weights[flg]
                if DEBUG and LOGIC_TEST:
                    logger.info(f'BATCH_IDX: {i} ORDERING_ACCOUNT')
            if (ba := try_decrypt_and_load(key, cipher_ba)) is not None:
                flg = self._get_flag(ba)
                ret[i] += self.weights[flg + 13]
                if DEBUG and LOGIC_TEST:
                    logger.info(f'BATCH_IDX: {i} BENEFICIARY_ACCOUNT')
        ret = quantize([ret], CLIP_RANGE, TARGET_RANGE)[0]
        masked_ret = masking(server_rnd, ret, self.cid, self.shared_seed_dict)
        if DEBUG:
            logger.info(f'client {self.cid}: masking offset = {server_rnd}')
        return [masked_ret], 0, {}

    def stage2(self, server_rnd, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        return [], 0, {}


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
    if cid == "swift":
        logger.info("Initializing SWIFT client for {}", cid)
        return TestSwiftClient(
            cid,
            df_pth=data_path,
            client_dir=client_dir,
            preds_format_path=preds_format_path,
            preds_dest_path=preds_dest_path,
        )
    else:
        logger.info("Initializing bank client for {}", cid)
        return TestBankClient(cid, df_pth=data_path, client_dir=client_dir)


class TestStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for test."""

    def __init__(self, server_dir: Path):
        self.server_dir = server_dir
        self.public_keys_dict = {}
        self.fwd_dict = {}
        self.agg_grad = np.zeros(26)
        self.stage = 0
        self.label = np.array(0)
        self.weights = np.array(0)
        self.logit = np.array(0)
        self.preds = np.array(0)
        self.encrypted_batch: Dict[str, bytes] = {}
        self.cached_banklsts = []
        self.rnd_cnt = 0
        super().__init__()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return empty_parameters()

    def __configure_round(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        cid_dict: Dict[str, ClientProxy] = client_manager.all()
        config_dict = {"round": server_round}
        logger.info(f"[START] server round {server_round}")
        if DEBUG:
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
            if DEBUG:
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
            fit_ins = FitIns(parameters=ndarrays_to_parameters([self.agg_grad]), config=config_dict)
            ins_lst = [(cid_dict['swift'], fit_ins)]
        elif self.stage == 1:
            logger.info(f"stage 1: broadcasting model weights and encrypted batch to bank clients")
            # broadcast the weights and the encrypted batch to all bank clients
            config_dict = config_dict | self.encrypted_batch
            fit_ins = FitIns(parameters=ndarrays_to_parameters([self.weights]), config=config_dict)
            ins_lst = [(proxy, fit_ins) for cid, proxy in cid_dict.items() if cid != 'swift']
        elif self.stage == 2:
            logger.info(f"stage 2: send y_pred to swift")
            fit_ins = FitIns(parameters=ndarrays_to_parameters([self.preds]), config=config_dict)
            ins_lst = [(cid_dict['swift'], fit_ins)]
        else:
            raise AssertionError("Stage number should be 0, 1, or 2")

        return ins_lst

    def __aggregate_round(
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
            masked_wx, weights = parameters_to_ndarrays(results[0][1].parameters)
            self.logit = masked_wx
            self.weights = weights
            # broadcast to all bank clients
            self.encrypted_batch = results[0][1].metrics
            logger.info(f"server received encrypted batch")
            # if server_round == self.num_rnds, do nothing
            pass
        elif self.stage == 1:
            # receive masked results from all bank clients
            for proxy, res in results:
                masked_res = parameters_to_ndarrays(res.parameters)[0]
                self.logit += masked_res
            self.logit &= 0xffffffff
            self.logit = reverse_quantize([self.logit], CLIP_RANGE, TARGET_RANGE)[0]
            self.logit -= len(results) * CLIP_RANGE
            if DEBUG:
                logger.info(f'server: reconstructed logits = {self.logit}')

            tmp = np.exp(-self.logit)
            y_pred = 1. / (1. + tmp)
            self.preds = y_pred
            pass
        elif self.stage == 2:
            pass
        else:
            raise AssertionError("Stage number should be 0, 1, or 2")
        self.stage = (self.stage + 1) % 3
        return None, {}

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.rnd_cnt += 1
        return self.__configure_round(self.rnd_cnt, parameters, client_manager)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        return self.__aggregate_round(self.rnd_cnt, results, failures)

    def configure_evaluate(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if server_round == 3:
            logger.info('Test Strategy: skip last eval round!')
            return []
        self.rnd_cnt += 1
        ins_lst = self.__configure_round(self.rnd_cnt, parameters, client_manager)
        return [(proxy, EvaluateIns(ins.parameters, ins.config)) for proxy, ins in ins_lst]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if server_round == 3:
            logger.info('Test Strategy: skip last eval round!')
            return None, {}

        def convert(res: EvaluateRes) -> FitRes:
            params = bytes2pyobj(res.metrics.pop('parameters'))
            return FitRes(res.status, params, res.num_examples, res.metrics)

        results = [(proxy, convert(eval_res)) for proxy, eval_res in results]
        return self.__aggregate_round(self.rnd_cnt, results, failures)

    def evaluate(self, server_round, parameters):
        """Not running any centralized evaluation."""
        return None


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
    test_strategy = TestStrategy(server_dir=server_dir)
    # setup rounds = 2 / 2 = 1
    # predict rounds = 3 / 2 = 1.5 (stage 0, 1, 2)
    # num_rounds = 1 + 1.5 = 2.5
    num_rounds = 3
    return test_strategy, num_rounds
