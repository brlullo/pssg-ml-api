#!/usr/bin/env Rscript
# One-time extraction of the VUMC HGT complication model (rms::lrm fit) into
# plain-text intermediates, assembled into model.json by build_model_json.py.
#
# Base R only (no rms/jsonlite needed — an rms fit object is just a list).
#
# Usage:
#   Rscript scripts/extract_hgt_model.R <path/to/fit0.rds> <output_dir>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: extract_hgt_model.R <fit.rds> <outdir>")
rds_path <- args[1]
outdir <- args[2]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

fit <- readRDS(rds_path)
cf <- fit$coefficients
V <- fit$var
d <- fit$Design

fmt <- function(x) sprintf("%.17g", x)

# --- sanity checks ---
stopifnot(identical(rownames(V), names(cf)))
stopifnot(identical(colnames(V), names(cf)))
stopifnot(fit$non.slopes == 3)

# --- coefficients (full precision) ---
write.csv(
  data.frame(name = names(cf), value = fmt(cf), stringsAsFactors = FALSE),
  file.path(outdir, "coef.csv"), row.names = FALSE
)

# --- covariance matrix (full precision, row-major with names) ---
Vc <- matrix(fmt(V), nrow = nrow(V))
write.csv(
  data.frame(name = rownames(V), Vc, stringsAsFactors = FALSE),
  file.path(outdir, "var.csv"), row.names = FALSE
)

# --- design info: one row per predictor; values joined with | ---
rows <- lapply(seq_along(d$name), function(i) {
  nm <- d$name[i]
  kind <- d$assume[i]
  p <- d$parms[[nm]]
  vals <- if (kind == "rcspline") paste(fmt(p), collapse = "|")
          else if (kind == "category") paste(p, collapse = "|")
          else ""  # interactions: structure recovered from coefficient names
  data.frame(name = nm, kind = kind, values = vals, stringsAsFactors = FALSE)
})
write.csv(do.call(rbind, rows), file.path(outdir, "design.csv"), row.names = FALSE)

# --- misc scalars ---
writeLines(c(
  paste0("ylevels=", paste(names(fit$freq), collapse = "|")),
  paste0("yfreq=", paste(as.integer(fit$freq), collapse = "|")),
  paste0("non_slopes=", fit$non.slopes),
  paste0("n_obs=", fit$stats[["Obs"]]),
  paste0("c_stat=", fmt(fit$stats[["C"]])),
  paste0("n_impute=", if (!is.null(fit$n.impute)) fit$n.impute else NA)
), file.path(outdir, "misc.txt"))

cat("Extraction complete. Files written to", outdir, "\n")
