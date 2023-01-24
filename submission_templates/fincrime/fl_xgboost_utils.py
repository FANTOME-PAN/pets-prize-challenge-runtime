from pathlib import Path
from loguru import logger
import pandas as pd
from xgboost import XGBClassifier
from .model_centralized import pre_process_swift, pre_process_swift_test, \
    transform_and_normalized_X_train, transform_and_normalized_Y, transform_and_normalized_X_test, get_X_swift


def fit_swift(
        swift_data: pd.DataFrame,
        model_dir: Path,
) -> XGBClassifier:
    """Function that fits your XGBoost on the SWIFT training data and saves
    your XGBoost to disk in the provided directory.
    Args:
        swift_data (pd.DataFrame): CSV data file for the SWIFT transaction
            dataset.
        model_dir (Path): Path to a directory that is constant between the train
            and test stages. You must use this directory to save and reload
            your trained model between the stages.
    Returns: XGBClassifier, Probabilities
    """

    logger.info("read the table")
    swift_train = swift_data
    swift_train["Timestamp"] = swift_train["Timestamp"].astype("datetime64[ns]")

    logger.info("pre-processing the swift dataset")
    swift_train = pre_process_swift(swift_train, model_dir)
    columns_to_drop = [
        "UETR",
        "Sender",
        "Receiver",
        "TransactionReference",
        "OrderingAccount",
        "OrderingName",
        "OrderingStreet",
        "OrderingCountryCityZip",
        "BeneficiaryAccount",
        "BeneficiaryName",
        "BeneficiaryStreet",
        "BeneficiaryCountryCityZip",
        "SettlementDate",
        "SettlementCurrency",
        "InstructedCurrency",
        "Timestamp",
        "sender_hour",
        "receiver_currency",
        "sender_receiver",
    ]

    swift_train = swift_train.drop(columns_to_drop, axis=1)

    # combine_train = swift_train.reset_index().rename(columns={'index': 'MessageId'})  # Not sure about this line
    # cols = ['SettlementAmount', 'InstructedAmount', 'Label', 'hour',
    #         'sender_hour_freq', 'currency_freq', 'currency_amount_average', 'sender_receiver_freq',
    #         'receiver_currency_amount_average', 'sender_receiver_freq']
    # swift_train = swift_train[cols]
    # combine_train = combine_swift_and_bank_new(swift_train, bank_train)

    logger.info("Get X_train and Y_train")
    X_train = transform_and_normalized_X_train(swift_train, model_dir)
    Y_train = transform_and_normalized_Y(swift_train)

    logger.info("Fit SWIFT XGBoost")

    X_train_swift = get_X_swift(X_train)
    xgb = XGBClassifier(n_estimators=100, max_depth=7, base_score=0.01,
                        learning_rate=0.1)  # These parameters can be tuned
    xgb.fit(X_train_swift, Y_train)

    logger.info("Save XGBoost")
    xgb.save_model(model_dir / "xgb.json")  # Rename this if you like

    logger.info("Get probability from XGBoost")
    pred_proba_xgb_train = xgb.predict_proba(X_train_swift)[:, 1]

    return xgb, pred_proba_xgb_train


def test_swift(swift_data: pd.DataFrame, model_dir: Path):
    logger.info("read the table")
    swift_test = swift_data
    swift_test["Timestamp"] = swift_test["Timestamp"].astype("datetime64[ns]")

    logger.info("pre-processing the swift dataset")
    swift_test = pre_process_swift_test(swift_test, model_dir)

    columns_to_drop = [
        "UETR",
        "Sender",
        "Receiver",
        "TransactionReference",
        "OrderingAccount",
        "OrderingName",
        "OrderingStreet",
        "OrderingCountryCityZip",
        "BeneficiaryAccount",
        "BeneficiaryName",
        "BeneficiaryStreet",
        "BeneficiaryCountryCityZip",
        "SettlementDate",
        "SettlementCurrency",
        "InstructedCurrency",
        "Timestamp",
        "sender_hour",
        "receiver_currency",
        "sender_receiver",
    ]

    swift_test = swift_test.drop(columns_to_drop, axis=1)

    # combine_test = swift_test.reset_index().rename(columns={'index': 'MessageId'})  # Not sure about this line
    # # combine_train = combine_swift_and_bank_new(swift_train, bank_train)
    # cols = ['SettlementAmount', 'InstructedAmount', 'hour', 'sender_hour_freq',
    #         'currency_freq', 'currency_amount_average', 'sender_receiver_freq',
    #         'receiver_currency_amount_average', 'sender_receiver_freq']
    # combine_test = combine_test[cols]

    logger.info("Get X_test")
    X_test = transform_and_normalized_X_test(swift_test, model_dir, if_exist_y=False)

    logger.info("Run SWIFT XGBoost")

    X_test_swift = get_X_swift(X_test)
    xgb = XGBClassifier(n_estimators=100, max_depth=7, base_score=0.01,
                        learning_rate=0.1)  # These parameters can be tuned

    logger.info("Load XGBoost")
    xgb.load_model(model_dir / "xgb.json")  # Rename this if you like

    logger.info("Get probability from XGBoost")
    pred_proba_xgb = xgb.predict_proba(X_test_swift)[:, 1]

    return xgb, pred_proba_xgb
