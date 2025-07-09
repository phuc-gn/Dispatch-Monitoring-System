# !/bin/bash

SRC_DIR="data/Classification"
DEST_DIR="data/Classification_split"
SPLIT=0.8

mkdir -p "$DEST_DIR"

for TYPE in dish tray; do
  for CLASS in empty kakigori not_empty; do
    SRC_CLASS_DIR="$SRC_DIR/$TYPE/$CLASS"
    DEST_TRAIN="$DEST_DIR/$TYPE/train/$CLASS"
    DEST_VAL="$DEST_DIR/$TYPE/val/$CLASS"

    mkdir -p "$DEST_TRAIN" "$DEST_VAL"

    IMAGES=($(ls "$SRC_CLASS_DIR"))
    TOTAL=${#IMAGES[@]}
    TRAIN_COUNT=$(( TOTAL * 80 / 100 ))

    for ((i=0; i<TOTAL; i++)); do
      IMG="${IMAGES[$i]}"
      if [ "$i" -lt "$TRAIN_COUNT" ]; then
        cp "$SRC_CLASS_DIR/$IMG" "$DEST_TRAIN/"
      else
        cp "$SRC_CLASS_DIR/$IMG" "$DEST_VAL/"
      fi
    done
  done
done

SRC_DIR="data/Detection"

cat > "$SRC_DIR/dataset_edit.yaml" << EOF
path: .
train: train/images
val: val/images

nc: 2
names:
  0: dish
  1: tray
EOF