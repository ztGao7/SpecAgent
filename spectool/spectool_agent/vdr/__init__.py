"""
VDR（Vision-DeepResearch）Agent 栈 — 隔离的视觉深度研究模块。

VDR-Bench 由 Vision-DeepResearch 模型服务，其原生工具调用/答案格式
与 Qwen3.5 系列模型不同。为避免相互干扰，VDR 路径拥有自己独立的：
  - 提示词（prompts_vdr）
  - 解析器（parsers_vdr，支持 XML 格式的工具调用）
  - Draft 模型封装（draft_model_vdr）
  - Agent 循环（agent_vdr 中的 VDRSpeculativeAgentLoop）

格式无关的组件（VLLMModel 传输层、QJ/一致性辅助函数、图片辅助函数、工具注册表、
  DraftPool、AgentConfig 等）仍然从共享模块导入。
"""
