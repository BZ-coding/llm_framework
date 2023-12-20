# LLM_FrameWork

我们希望通过编写自己的大模型框架的方式，不光有个能用的、好用的框架，同时也尽可能把大模型中的坑和trick自己走一遍，提升个人水平。

-----------------------

## Infer
### LLama-7B:
```
<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.
 ### Instruction:
 Give me three healthy life tips.
 ### Response:
 1) Eat more fruits and vegetables, especially leafy greens like spinach or kale! They are full of vitamins A & C which help your body fight off infections as well as keep you looking young by keeping skin smooth and supple. Also try to eat foods with high fiber content such as whole grain bread instead of white flour products because they will make you feel less hungry between meals so you won't overeat later on when it comes time for dinner (or lunch).  2) Exercise regularly - at least thirty minutes per day five days out of seven if possible but even just walking around outside can be beneficial since fresh air helps clear away pollutants from our bodies. If you have access to exercise equipment then use them too; however don’t forget about simple exercises done without any specialized tools – push-ups work wonders for strengthening muscles all throughout one’s upper torso while sit ups tone abdominal areas nicely.   3) Get enough sleep every night no matter what else may need doing during those hours. Sleeping allows us to recharge ourselves physically and mentally after being awake through most of each twenty four hour period. It also gives our immune systems some much needed rest allowing their cells to regenerate themselves faster than usual thereby making sure we stay free of illness longer periods of times.</s>
```

### LLama-7B + LoRA(alpaca):
```
<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.
 ### Instruction:
 Give me three healthy life tips.
 ### Response:
 1) Eat plenty of fruits and vegetables to get all your essential vitamins, minerals, antioxidants, fiber, and other nutrients needed for good overall wellness; 2) Exercise regularly by doing activities such as walking or running at least thirty minutes per day five days out of seven in order to keep fit and maintain optimal physical conditioning; 3) Get enough sleep every night so you can recharge both physically and mentally throughout each day.</s>
```

-----------------------

## 单卡
- [x] 指定tokenizer的pad_token_id
- [x] 推理脚本
- [x] 使用transformer trainer api的lora微调脚本
- [x] 使用pytorch+accelerator的lora微调脚本
- [ ] loss mask
- [ ] 推理速度（编译）
- [ ] 推理支持stream模式
- [ ] 模型编译
- [ ] 模型量化

-----------------------

## 单机多卡 & 多机多卡
- [ ] megatron
- [ ] deepspeed
