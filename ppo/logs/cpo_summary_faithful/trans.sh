#!/bin/bash

# 脚本功能：遍历当前目录下所有.csv文件，
# 并将表头中的'obtained_summary'替换为'obtained_score1'，
# 'obtained_faithful'替换为'obtained_score2'。

# 计数器，用于统计处理了多少文件
count=0

# 遍历所有.csv文件
for file in *.csv; do
  # 检查文件是否存在，防止没有匹配项时出错
  if [ -f "$file" ]; then
    echo "正在处理: $file"
    
    # 使用sed进行原地替换(in-place)
    # -i.bak 会创建一个.bak后缀的备份文件，这是一个好习惯！
    # '1s/.../.../g' 表示只在第1行进行替换
    # 用分号(;)连接两个替换命令
    sed -i.bak '1s/obtained_summary/obtained_score1/g; 1s/obtained_faithful/obtained_score2/g' "$file"
    
    count=$((count+1))
  fi
done

if [ $count -eq 0 ]; then
  echo "未找到任何 .csv 文件。"
else
  echo "处理完成！共修改了 $count 个文件。"
  echo "原始文件已备份为 .csv.bak 文件。"
fi

# 如果你确认修改无误，可以运行下面的命令删除所有备份文件
# read -p "确认修改无误并删除所有备份文件吗？(y/n) " -n 1 -r
# echo
# if [[ $REPLY =~ ^[Yy]$ ]]; then
#   echo "正在删除所有 .bak 文件..."
#   rm -- *.csv.bak
#   echo "备份文件已删除。"
# fi
