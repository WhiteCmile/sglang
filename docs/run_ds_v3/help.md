# 一些需要注意的地方

1. DeepSeek-V3 的 PATH 保存在 `/share/zhaotianlang/scripts/set_model.sh` 中，直接
```
source /share/zhaotianlang/scripts/set_model.sh
echo $DEEPSEEK_V3_PATH
```
就可以拿到
2. 两个 node 在跑命令之前记得 `ulimit -l unlimited`