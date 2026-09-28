from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_forecast_program():
    path = ROOT / "code" / "Tables_04_05_06_forecast_performance.py"
    spec = importlib.util.spec_from_file_location("forecast_performance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    program = load_forecast_program()
    table4 = program.build_table_04()
    table5 = program.build_table_05()
    table6 = program.build_table_06()
    markets = table4.check_inputs()
    if table5.check_inputs() != markets or table6.check_inputs() != markets:
        raise ValueError("The three tables must use the same markets")
    predictions4, predictions5, predictions6, layer_predictions = {}, {}, {}, {}
    for runners, target in (
        (table4.MODEL_RUNNERS, predictions4),
        (table5.AUXILIARY_RUNNERS + table5.MODEL_RUNNERS, predictions5),
        (table6.STANDARD_RUNNERS, predictions6),
    ):
        for key, label, runner in program.tqdm(runners):
            result = runner()
            program.validate_forecasts(label, result, markets)
            target[key] = result
    for key, runner in program.tqdm(table4.LAYER_RUNNERS):
        result = runner()
        program.validate_forecasts(key, result, markets)
        layer_predictions[key] = result
    shared = program.compute_shared_drop_dates(
        table4, predictions4, predictions5, predictions6, layer_predictions
    )
    rows = [
        (market, pd.Timestamp(date).strftime("%Y-%m-%d"), 1)
        for market in sorted(shared)
        for date in sorted(shared[market])
    ]
    output = pd.DataFrame(rows, columns=["market", "date", "drop_for_all_models"])
    output_dir = ROOT / "results" / "generated" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "qlike_intermediate.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
