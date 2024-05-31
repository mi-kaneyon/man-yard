#!/bin/bash

# ログファイルの設定
LOGFILE="/home/educate/startdemo.log"

# ログの初期化
echo "Script started at $(date)" > $LOGFILE

# Condaの初期化
echo "Initializing conda..." >> $LOGFILE
source /home/educate/anaconda3/etc/profile.d/conda.sh >> $LOGFILE 2>&1
if [ $? -ne 0 ]; then
  echo "Failed to source conda.sh" >> $LOGFILE
  exit 1
fi

# 仮想環境をアクティベート
echo "Activating conda environment 'ldm'..." >> $LOGFILE
conda activate ldm >> $LOGFILE 2>&1
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment 'ldm'" >> $LOGFILE
  exit 1
fi

# Web UIのディレクトリに移動
echo "Changing directory to /home/educate/path/to/StableDiffusionWebUI..." >> $LOGFILE
cd /home/educate/path/to/StableDiffusionWebUI >> $LOGFILE 2>&1
if [ $? -ne 0 ]; then
  echo "Failed to change directory to /home/educate/path/to/StableDiffusionWebUI" >> $LOGFILE
  exit 1
fi

# Web UIを起動
echo "Starting Web UI..." >> $LOGFILE
python launch.py >> $LOGFILE 2>&1
if [ $? -ne 0 ]; then
  echo "Failed to start Web UI" >> $LOGFILE
  exit 1
fi

# 完了メッセージ
echo "Script completed at $(date)" >> $LOGFILE
