# 🎬 AnimePick · 番剧推荐

🌐 **在线访问:<https://sateriajiaying.github.io/anime-pick/>**

(隔壁还有我的图片收藏:<https://sateriajiaying.github.io/anime-gallery/>)

一个纯前端的交互式动画推荐系统:勾选 20 部看过的番 → 自适应 12~40 轮快问快答 → 分层推荐(热门精选 + 小众之选),附推荐理由、同好短评与兴趣画像。

## 功能

| 页面 | 内容 |
|---|---|
| 首页 | 规则说明 + 数据面板 + 开始测试 |
| 推荐流程 | 20 锚点勾选 → 六档问答(没看过/没兴趣/一般/喜欢/不喜欢/超喜欢)→ 结果页(画像/推荐/看过了重排) |
| 💘 发现 | 选一部让你心动的番,从口味坐标/共同深爱/标签/同好观众四维度推荐 |
| 📊 统计 | 口味地图(t-SNE)、意外动画对、口碑实验室、64 类观众群、年代与载体、数据自画像 |
| 📜 方法 | 论文式 Methods:数据、模型、评估、复现指南 |

## 数据与模型

- 语料:Bangumi 番组计划公开数据(wiki 归档 + 官方 API 用户评分 306,368 条、2,021 用户)
- 模型:64 维隐因子(BPR 排序 + ALS 提问)+ 64 个品味原型 + 超喜欢共现图,int8 量化约 1.5MB
- 全部推理在浏览器本地完成,无后端、不收集个人信息

## 复现

完整管线见 [`pipeline/`](pipeline/)(爬虫、训练、打包脚本 + 说明)。

## 反馈

- [GitHub Issues](https://github.com/SATERIAJIAYING/anime-pick/issues)
- 邮件:Xingqaq@qq.com

## 许可

- **代码**:[0BSD](LICENSE)(零要求,可任意使用、修改、商用)
- **数据与文案**:[CC0 1.0](LICENSE-DATA)(放弃一切权利,无需署名)
- 数据内容源自 [Bangumi 番组计划](https://bangumi.github.io/api/),内容权利归 Bangumi 及原作者;统计产物仅含聚合信息,不含单个用户标识
