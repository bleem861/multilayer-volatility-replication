suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(zoo))
suppressPackageStartupMessages(library(ConnectednessApproach))

script_path <- NULL
for (frame in sys.frames()) {
  if (!is.null(frame$ofile)) script_path <- frame$ofile
}
if (is.null(script_path)) {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) script_path <- sub("^--file=", "", file_arg[1])
}
if (is.null(script_path) && requireNamespace("rstudioapi", quietly = TRUE)) {
  if (rstudioapi::isAvailable()) {
    editor_path <- rstudioapi::getSourceEditorContext()$path
    if (nzchar(editor_path)) script_path <- editor_path
  }
}
if (!is.null(script_path)) {
  root <- dirname(dirname(normalizePath(script_path, mustWork = TRUE)))
} else {
  root <- normalizePath(getwd(), mustWork = TRUE)
  if (basename(root) == "code") root <- dirname(root)
}
if (!file.exists(file.path(root, "data", "raw", "Return.csv")) && interactive()) {
  selected_root <- utils::choose.dir(caption = "Select the proj folder containing data and code")
  if (!is.na(selected_root)) root <- selected_root
}
input_path <- file.path(root, "data", "raw", "Return.csv")
if (!file.exists(input_path)) stop("Required input not found: ", input_path)
output_dir <- file.path(root, "results", "generated", "spillover")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

volatility_raw <- read_csv(input_path, show_col_types = FALSE)
dates <- as.Date(volatility_raw[[1]], format = "%Y/%m/%d")
x <- as.data.frame(volatility_raw[, 2:19])
x[] <- lapply(x, function(col) as.numeric(col))
volatility <- zoo(x, order.by = dates)

invisible(capture.output(
  volatility_dynamic <- ConnectednessApproach::ConnectednessApproach(
    volatility,
    nlag = 1,
    nfore = 200,
    model = "TVP-VAR",
    connectedness = "Frequency",
    window.size = NULL,
    VAR_config = list(TVPVAR = list(
      kappa1 = 0.99, kappa2 = 0.96, prior = "BayesPrior", gamma = 0.01
    )),
    Connectedness_config = list(FrequencyConnectedness = list(
      partition = c(pi + 0.00001, pi / 5, pi / 20, 0),
      generalized = TRUE,
      scenario = "ABS"
    ))
  )
))
TCI_df <- data.frame(TCI = volatility_dynamic[["TCI"]])
write.csv(TCI_df, file = file.path(output_dir, "TCI_data_BK-TVP.csv"))
