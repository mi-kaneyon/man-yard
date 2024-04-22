# ROWVAMR Rakuten AI 7B Rakuten Japanese LLM
- Base on Hugginface project
- Install transformers
- need several requirement module for LLM.

# ROW VRAM chatbot script 

```
python rakuchat_rawvram.py

```

# customized auto chatbot script 
- Input first sentence and script proceed converation 

```
python respchat.py

```


## Example 

```
チャットボットを開始します。何か質問はありますか？(終了するには 'exit' と入力してください)
USER: 日本から一番近い国で飛行機で何時間ですか？
ASSISTANT: 日本から一番近い国で飛行機で何時間ですか？

日本から一番近い国で、飛行機で何時間かかるか答える。

答え

韓国、ロシア（サハ共和国）、中国、北朝鮮、台湾、フィリピン、マレーシア、インドネシア、東ティモール、オーストラリア、パプアニューギニア、ソロモン諸島、バヌアツ、ニュージーランド、フランス（ニューカレドニア）、アメリカ（ミッドウェー島）、イギリス（ピトケアン島）、キルギス、ウズベキスタン、カザフスタン、トルクメニスタン、アフガニスタン、パキスタン、インド、スリランカ、モルディブ、アラビア首長国連邦、オマーン、イエメン、サウジアラビア、レバノン、イスラエル、エジプト、スーダン、エリトリア、エチオピア、ジブチ、ソマリア、モーリタニア、モロッコ、スペイン（セウタ）、スペイン（メリリャ）、アルジェリア、チュニジア、リビア、マリ、ニジェール、チャド、中央アフリカ、カメルーン、コンゴ共和国、コンゴ民主共和国、赤道ギニア、ガボン、アンゴ
USER: Killed


```

# Citation

@misc{rakutengroup2024rakutenai7b,
      title={RakutenAI-7B: Extending Large Language Models for Japanese}, 
      author={{Rakuten Group, Inc.} and Aaron Levine and Connie Huang and Chenguang Wang and Eduardo Batista and Ewa Szymanska and Hongyi Ding and Hou Wei Chou and Jean-François Pessiot and Johanes Effendi and Justin Chiu and Kai Torben Ohlhus and Karan Chopra and Keiji Shinzato and Koji Murakami and Lee Xiong and Lei Chen and Maki Kubota and Maksim Tkachenko and Miroku Lee and Naoki Takahashi and Prathyusha Jwalapuram and Ryutaro Tatsushima and Saurabh Jain and Sunil Kumar Yadav and Ting Cai and Wei-Te Chen and Yandi Xia and Yuki Nakayama and Yutaka Higashiyama},
      year={2024},
      eprint={2403.15484},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
