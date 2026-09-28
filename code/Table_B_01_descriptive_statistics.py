from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_FILE = ROOT / "results" / "generated" / "Table_B_01_descriptive_statistics.xlsx"


def significance(value, p_value):
    if p_value < 0.01:
        stars = "***"
    elif p_value < 0.05:
        stars = "**"
    elif p_value <= 0.10:
        stars = "*"
    else:
        stars = ""
    rounded = round(value, 4)
    text = str(Decimal(str(rounded)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return f"{text} {stars}" if stars else text


def load_data(filename, scale):
    data = pd.read_csv(RAW_DIR / filename)
    data["date"] = pd.to_datetime(data["date"])
    value_columns = data.columns.drop("date")
    data[value_columns] = data[value_columns].apply(pd.to_numeric, errors="coerce") * scale
    data = data.dropna().set_index("date")
    data.columns = [column.lstrip(".") for column in data.columns]
    return data


def calculate_statistics(data):
    rows = []
    for market in data.columns:
        values = data[market]
        jb_result = stats.jarque_bera(values)
        adf_result = adfuller(values)
        lb_result = acorr_ljungbox(values, lags=10, return_df=True)
        rows.append(
            [
                float(np.mean(values)),
                float(np.std(values)),
                float(stats.skew(values)),
                float(stats.kurtosis(values, fisher=False)),
                significance(float(jb_result.statistic), float(jb_result.pvalue)),
                significance(float(adf_result[0]), float(adf_result[1])),
                significance(float(lb_result.iloc[0, 0]), float(lb_result.iloc[1, -1])),
            ]
        )
    return pd.DataFrame(
        rows,
        index=data.columns,
        columns=["Mean", "Std.", "Skew", "Kurt", "JB test", "ADF", "LB-Q(10)"],
    )


def write_workbook(rv_statistics, return_statistics):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Table B.1"
    sheet.sheet_view.showGridLines = False
    sheet.merge_cells("B1:H1")
    sheet.merge_cells("I1:O1")
    sheet["B1"] = "Realized volatility"
    sheet["I1"] = "Daily returns"
    headers = [
        "Index",
        "Mean (x10000)",
        "Std. (x10000)",
        "Skew",
        "Kurt",
        "JB test",
        "ADF",
        "LB-Q(10)",
        "Mean (x100)",
        "Std. (x100)",
        "Skew",
        "Kurt",
        "JB test",
        "ADF",
        "LB-Q(10)",
    ]
    for column, header in enumerate(headers, start=1):
        sheet.cell(row=2, column=column, value=header)
    for row_number, market in enumerate(rv_statistics.index, start=3):
        sheet.cell(row=row_number, column=1, value=market)
        values = rv_statistics.loc[market].tolist() + return_statistics.loc[market].tolist()
        for column, value in enumerate(values, start=2):
            sheet.cell(row=row_number, column=column, value=value)
    thin = Side(style="thin", color="000000")
    for cell in sheet[1]:
        cell.border = Border(top=thin, bottom=thin)
    for cell in sheet[2]:
        cell.border = Border(bottom=thin)
    for cell in sheet[sheet.max_row]:
        cell.border = Border(bottom=thin)
    for row in sheet.iter_rows(min_row=1, max_row=sheet.max_row, min_col=1, max_col=15):
        for cell in row:
            cell.font = Font(name="Times New Roman", size=10)
            cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in range(3, sheet.max_row + 1):
        for column in list(range(2, 6)) + list(range(9, 13)):
            sheet.cell(row=row, column=column).number_format = "0.000"
    widths = [11, 15, 14, 10, 10, 19, 13, 15, 15, 14, 10, 10, 17, 13, 15]
    for column, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.row_dimensions[1].height = 20
    sheet.row_dimensions[2].height = 30
    sheet.freeze_panes = "B3"
    workbook.save(OUTPUT_FILE)


def main():
    rv_data = load_data("RV.csv", 10000)
    return_data = load_data("Return.csv", 100)
    common_dates = rv_data.index.intersection(return_data.index)
    rv_data = rv_data.loc[common_dates]
    return_data = return_data.loc[common_dates]
    rv_statistics = calculate_statistics(rv_data)
    return_statistics = calculate_statistics(return_data)
    write_workbook(rv_statistics, return_statistics)


if __name__ == "__main__":
    main()
