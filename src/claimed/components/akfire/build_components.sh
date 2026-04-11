#!/usr/bin/env bash
# build_components.sh – compiles and unpacks all operators to version 0.1.0
# usage:  chmod +x build_components.sh && ./build_components.sh

set -euo pipefail

VERSION="0.1.0"
COMPILER="c3_create_containerless_operator"
OUTDIR="claimed-operators"

operators=(
	"operators/create_training_zarr.py"
	"operators/train_logistic.py"
	"operators/logistic_prediction.py"
	"operators/training_xgboost.py"
	"operators/xgboost_prediction.py"
	"operators/optimize_xgb_hyperparameters_from_df.py"
)

mkdir -p "$OUTDIR"

for op in "${operators[@]}"; do
	base="$(basename "$op" .py)"
	echo ">> Compiling $base:$VERSION"
	"$COMPILER" -v "$VERSION" "$op"

	zip="$OUTDIR/${base}:$VERSION.zip"         # default output from compiler
	dest="$OUTDIR/${base}.$VERSION"            # destination folder
	echo "   Unpacking to $dest"
	rm -rf "$dest"
	unzip -q "$zip" -d "$dest"
done

echo "Components version $VERSION ready in $OUTDIR/"
