# ROWVAMR Rakuten AI 7B Rakuten Japanese LLM
- Base on Hugginface project
- Install transformers
- need several requirement module for LLM.

# person expression image answer

```
python person_expression.py

```
## 

> [!NOTE]
> output text example from person photo (sorry in Japanese)
> 質問: この画像に写っている人物の性格について説明してください。
>
> 回答: この画像に写っている人物の性格について説明してください。
> 
> この画像に写っている人物の性格について説明してください。
> この写真の人物は、真剣な表情でありながら目尻に微笑みを浮かべている。
> そのため、この写真の人物は笑いを抑えるのが苦手で、常に笑顔を絶やすことなく過ごしているようである。
> この写真の人物は、楽しいことをしたときに自然と笑顔になるのであり、怒ったり悔しがったりはしないようである。
> この写真の人物は、いつも笑顔であり、いつも人を笑わせているようである。
> この写真の人物は、楽しいことをすることが好きなようである。
> この写真の人物は、いつも自分のことを悔いるよりも笑うほう


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
