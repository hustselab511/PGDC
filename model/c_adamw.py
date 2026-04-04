# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import cast, Callable, Iterable, Tuple, Optional, Union, List, Dict, Any
import torch
import math
from torch import nn
from torch import Tensor
from torch.distributed._tensor import distribute_tensor, DTensor
from torch.optim.optimizer import Optimizer


def _multi_tensor_adam(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        grad_scale: Optional[Tensor] = None,
        found_inf: Optional[Tensor] = None,
        *,
        amsgrad: bool,
        has_complex: bool,
        beta1: Union[float, Tensor],
        beta2: Union[float, Tensor],
        lr: Union[float, Tensor],
        weight_decay: float,
        eps: float,
        maximize: bool,
        capturable: bool,
        differentiable: bool,
        decoupled_weight_decay: bool,
        cautious: bool = False,
):
    if not params:
        return

    # 验证输入参数有效性
    if isinstance(lr, Tensor):
        if not capturable:
            raise RuntimeError("lr as Tensor is only supported with capturable=True")
        if lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")

    for beta, name in [(beta1, "beta1"), (beta2, "beta2")]:
        if isinstance(beta, Tensor):
            if not capturable:
                raise ValueError(f"{name} as Tensor is only supported with capturable=True")
            if beta.numel() != 1:
                raise ValueError(f"Tensor {name} must be 1-element")

    # 验证设备兼容性
    if not torch.compiler.is_compiling() and capturable:
        supported_devices = {"cuda", "cpu"}  # 简化的支持设备列表
        for p, step in zip(params, state_steps):
            if p.device.type != step.device.type or p.device.type not in supported_devices:
                raise RuntimeError(
                    f"With capturable=True, all parameters and state steps must be on the same "
                    f"supported device (got {p.device.type} and {step.device.type})"
                )

    # 不支持的参数组合检查
    if grad_scale is not None or found_inf is not None:
        raise ValueError("grad_scale and found_inf are not supported in this implementation")
    if differentiable:
        raise ValueError("_multi_tensor_adam does not support autograd (differentiable=True)")

    # 处理学习率（转为标量）
    lr = lr.item() if isinstance(lr, Tensor) else lr

    # 按设备和数据类型分组张量以提高效率
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]
    )

    # 准备beta1的设备映射（如果是张量）
    beta1_device_map: Optional[Dict[torch.device, Tensor]] = None
    if isinstance(beta1, Tensor) and beta1.device.type != "cpu":
        beta1_device_map = {beta1.device: beta1}

    # 处理每个设备组
    for (
            device_params, device_grads, device_exp_avgs,
            device_exp_avg_sqs, device_max_exp_avg_sqs, device_state_steps
    ), _ in grouped_tensors.values():
        # 类型转换确保类型安全
        device_params = cast(List[Tensor], device_params)
        device_grads = cast(List[Tensor], device_grads)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs)
        device_state_steps = cast(List[Tensor], device_state_steps)

        device = device_params[0].device

        # 获取当前设备的beta1值
        current_beta1 = beta1
        if beta1_device_map is not None:
            if device not in beta1_device_map:
                beta1_device_map[device] = beta1.to(device=device, non_blocking=True)
            current_beta1 = beta1_device_map[device]

        # 处理复数参数
        if has_complex:
            view_args = [device_params, device_grads, device_exp_avgs, device_exp_avg_sqs]
            if amsgrad:
                view_args.append(cast(List[Tensor], device_max_exp_avg_sqs))
            _view_as_real(*view_args)

        # 处理最大化目标（梯度取反）
        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        # 更新步骤计数
        step_incr = torch.tensor(1.0, device=device) if device_state_steps[0].is_cpu else 1.0
        torch._foreach_add_(device_state_steps, step_incr)

        # 处理权重衰减
        if weight_decay != 0:
            if decoupled_weight_decay:
                # 解耦权重衰减：直接更新参数
                torch._foreach_mul_(device_params, 1 - lr * weight_decay)
            else:
                # 传统权重衰减：加入梯度
                weight_decay_factor = weight_decay if not maximize else -weight_decay
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay_factor)

        # 更新一阶矩估计 (exp_avg = beta1 * exp_avg + (1 - beta1) * grad)
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - current_beta1)

        # 更新二阶矩估计 (exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2)
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        if isinstance(beta2, Tensor):
            scaled_grads = torch._foreach_mul(device_grads, 1 - beta2)
            add_factor = 1.0
        else:
            scaled_grads = device_grads
            add_factor = 1 - beta2
        torch._foreach_addcmul_(device_exp_avg_sqs, scaled_grads, scaled_grads, add_factor)
        del scaled_grads  # 释放临时张量

        # 计算参数更新
        if capturable:
            # 可捕获模式下的偏差校正
            bias_correction1 = torch._foreach_pow(torch.as_tensor(beta1, device=device), device_state_steps)
            bias_correction2 = torch._foreach_pow(torch.as_tensor(beta2, device=device), device_state_steps)

            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            torch._foreach_neg_(bias_correction2)  # 1 - beta2^t

            # 计算步长和校正项
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)  # lr / (1 - beta1^t)
            step_size = bias_correction1

            torch._foreach_sqrt_(bias_correction2)
            bias_correction2_sqrt = bias_correction2

            # 处理AMSGrad
            if amsgrad:
                max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs)
                torch._foreach_maximum_(max_exp_avg_sqs, device_exp_avg_sqs)
                exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            # 应用偏差校正和epsilon
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

        else:
            # 标准模式下的偏差校正
            bias_correction1 = [1 - beta1 ** step.item() for step in device_state_steps]
            bias_correction2 = [1 - beta2 ** step.item() for step in device_state_steps]

            # 计算步长 (-lr / (1 - beta1^t))
            step_size = _stack_if_compiling([(-lr / bc) for bc in bias_correction1])
            bias_correction2_sqrt = [bc ** 0.5 for bc in bias_correction2]

            # 处理AMSGrad
            if amsgrad:
                max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs)
                torch._foreach_maximum_(max_exp_avg_sqs, device_exp_avg_sqs)
                exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            # 应用偏差校正和epsilon
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)

        # 应用参数更新 (支持谨慎更新模式)
        if cautious:
            # 计算掩码：基于梯度和动量的符号一致性
            mask = torch._foreach_mul(device_exp_avgs, device_grads)
            mask = [m.gt(0.0).to(dtype=e.dtype) for m, e in zip(mask, device_exp_avgs)]

            # 归一化掩码以避免过度抑制
            mean_mask = [m.mean().clamp(min=1e-3) for m in mask]
            mask = [m / mm for m, mm in zip(mask, mean_mask)]

            # 应用掩码到动量
            masked_exp_avg = torch._foreach_mul(device_exp_avgs, mask)
            torch._foreach_addcdiv_(device_params, masked_exp_avg, exp_avg_sq_sqrt,
                                    step_size if not capturable else 1.0)
        else:
            # 标准更新
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt,
                                    step_size if not capturable else 1.0)


# 辅助函数 (假设这些函数在原始代码中已定义)
def _view_as_real(*tensors: List[Tensor]) -> None:
    """将复数张量视为实数张量的视图"""
    for tensor_list in tensors:
        for i in range(len(tensor_list)):
            if tensor_list[i].is_complex():
                tensor_list[i] = torch.view_as_real(tensor_list[i]).flatten(1)


def _stack_if_compiling(tensors: List[Tensor]) -> Union[List[Tensor], Tensor]:
    """如果在编译模式下，将张量列表堆叠为单个张量"""
    if torch.compiler.is_compiling():
        return torch.stack(tensors)
    return tensors


def _get_value(tensor: Tensor) -> float:
    """获取张量的标量值"""
    return tensor.item()


def _get_scalar_dtype() -> torch.dtype:
    """获取标量张量的数据类型（兼容PyTorch版本）"""
    return torch.float32


class AdamW(Optimizer):
    """
    实现带有权重衰减解耦的Adam优化器（AdamW），并支持`cautious`更新模式（基于动量与梯度的符号一致性调整更新）。

    参数:
        params (`Iterable[nn.parameter.Parameter]`):
            需要优化的参数迭代器或定义参数组的字典。
        lr (`float`, 可选, 默认=0.001):
            学习率。
        betas (`Tuple[float, float]`, 可选, 默认=(0.9, 0.999)):
            Adam的动量参数(b1, b2)。
        eps (`float`, 可选, 默认=1e-6):
            数值稳定性的epsilon值。
        weight_decay (`float`, 可选, 默认=0.0):
            解耦的权重衰减系数。
        correct_bias (`bool`, 可选, 默认=True):
            是否应用偏差校正（BERT等模型可能使用False）。
        no_deprecation_warning (`bool`, 可选, 默认=False):
            是否禁用弃用警告。
        foreach (`bool`, 可选, 默认=False):
            是否使用多张量优化（提升效率）。
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        foreach: bool = False,
    ):
        # 弃用警告
        if not no_deprecation_warning:
            warnings.warn(
                "此AdamW实现已 deprecated，未来版本将移除。请使用PyTorch官方实现torch.optim.AdamW，"
                "或设置no_deprecation_warning=True禁用此警告。",
                FutureWarning,
            )

        # 参数有效性检查
        if lr < 0.0:
            raise ValueError(f"无效学习率: {lr}（必须>=0.0）")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"无效beta1参数: {betas[0]}（必须在[0.0, 1.0)范围内）")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"无效beta2参数: {betas[1]}（必须在[0.0, 1.0)范围内）")
        if eps < 0.0:
            raise ValueError(f"无效epsilon值: {eps}（必须>=0.0）")

        # 初始化优化器参数
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

        self.foreach = foreach  # 多张量模式开关
        self.init_lr = lr  # 初始学习率（用于可能的后续调整）

    def _init_group(
        self,
        group: Dict[str, Any],
        params_with_grad: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[torch.Tensor],
        state_steps: List[torch.Tensor],
    ) -> bool:
        """
        初始化参数组状态，收集需要优化的参数及相关状态变量。

        返回:
            bool: 是否包含复数类型参数
        """
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue  # 跳过无梯度的参数

            # 检查复数类型
            if torch.is_complex(p):
                has_complex = True

            params_with_grad.append(p)
            grads.append(p.grad)

            # 初始化参数状态（延迟初始化，节省内存）
            state = self.state[p]
            if len(state) == 0:
                # 步骤计数器（默认CPU上，减少设备通信开销）
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                # 一阶动量（与参数同类型、同设备）
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # 二阶动量
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            # 收集状态变量
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])

        return has_complex

    @torch.no_grad()  # 禁用梯度计算（优化步骤不需要）
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """
        执行单步优化。

        参数:
            closure (`Callable`, 可选): 重新计算损失的闭包函数

        返回:
            Optional[torch.Tensor]: 损失值（如果提供了closure）
        """
        # 计算损失（如果提供闭包）
        loss = None
        if closure is not None:
            with torch.enable_grad():  # 闭包需要梯度
                loss = closure()

        # 多张量模式（foreach）
        if self.foreach:
            self._step_foreach()
        # 普通模式（逐参数更新）
        else:
            self._step_single()

        return loss

    def _step_foreach(self) -> None:
        """多张量优化模式（高效批量处理参数）"""
        for group in self.param_groups:
            # 收集参数组信息
            params_with_grad: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            exp_avgs: List[torch.Tensor] = []
            exp_avg_sqs: List[torch.Tensor] = []
            max_exp_avg_sqs: List[torch.Tensor] = []  # AMSGrad用（当前未启用）
            state_steps: List[torch.Tensor] = []

            # 初始化组并获取复数类型信息
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            # 跳过无梯度的组
            if not params_with_grad:
                continue

            # 调用多张量Adam更新（复用之前定义的_multi_tensor_adam）
            beta1, beta2 = group["betas"]
            _multi_tensor_adam(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                max_exp_avg_sqs=max_exp_avg_sqs,
                state_steps=state_steps,
                grad_scale=None,
                found_inf=None,
                amsgrad=False,  # AdamW默认不启用AMSGrad
                has_complex=has_complex,  # 传递实际复数类型检查结果
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=False,  # 不最大化目标（默认最小化）
                capturable=False,  # 不启用可捕获模式
                differentiable=False,  # 优化步骤不可微
                decoupled_weight_decay=True,  # 启用AdamW的解耦权重衰减
                cautious=True,  # 启用cautious更新逻辑
            )

    def _step_single(self) -> None:
        """普通优化模式（逐参数更新）"""
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            correct_bias = group["correct_bias"]

            for p in group["params"]:
                if p.grad is None:
                    continue  # 跳过无梯度参数

                grad = p.grad  # 当前梯度
                state = self.state[p]  # 参数状态

                # 初始化状态（若未初始化）
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg = state["exp_avg"]  # 一阶动量
                exp_avg_sq = state["exp_avg_sq"]  # 二阶动量
                state["step"] += 1  # 步骤计数+1
                step = state["step"]

                # 1. 解耦权重衰减（AdamW核心特性）
                if weight_decay > 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                # 2. 更新一阶、二阶动量
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)  # m_t = β1*m_{t-1} + (1-β1)*g_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)  # v_t = β2*v_{t-1} + (1-β2)*g_t²

                # 3. 计算更新步长（含偏差校正）
                if correct_bias:
                    # 偏差校正：(1 - β^t)
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    # 步长调整：lr * sqrt(1-β2^t) / (1-β1^t)
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = lr

                # 4. Cautious更新：基于动量与梯度的符号一致性生成掩码
                if isinstance(grad, DTensor):
                    # 分布式张量（DTensor）处理
                    full_exp_avg = exp_avg.full_tensor()
                    full_grad = grad.full_tensor()
                    mask = (full_exp_avg * full_grad > 0).to(grad.dtype)  # 符号一致的位置为1
                    mask.div_(mask.mean().clamp_(min=1e-3))  # 归一化掩码（避免均值过小）
                    # 重新分布式掩码张量
                    mask = distribute_tensor(
                        mask,
                        device_mesh=grad.device_mesh,
                        placements=grad.placements
                    )
                else:
                    # 普通张量处理
                    mask = (exp_avg * grad > 0).to(grad.dtype)  # 符号一致的位置为1
                    mask.div_(mask.mean().clamp_(min=1e-3))  # 归一化（防止除零）

                # 5. 应用掩码调整更新量并更新参数
                denom = exp_avg_sq.sqrt().add_(eps)  # 二阶动量开方 + epsilon
                adjusted_update = (exp_avg * mask) / denom  # 调整后的梯度
                p.add_(adjusted_update, alpha=-step_size)  # 参数更新：p = p - step_size * adjusted_update


def test_c_adamw():
    import copy
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Linear(10, 1).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-2,
        foreach=True
    )
    # Dummy input and target
    x = torch.randn(16, 10, device=device)
    y = torch.randn(16, 1, device=device)

    # Forward
    y_pred = model(x)
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y)

    # Backward
    loss.backward()

    # Save params before step
    before = copy.deepcopy([p.detach().clone() for p in model.parameters()])

    # Step
    optimizer.step()
    optimizer.zero_grad()

    # Save params after step
    after = [p.detach().clone() for p in model.parameters()]

    # Print parameter changes
    for i, (b, a) in enumerate(zip(before, after)):
        diff = (b - a).abs().sum().item()
        print(f"Param {i} changed by: {diff:.6f}")
        assert diff > 0, f"Parameter {i} did not change!"

    print("✅ C-AdamW with foreach ran successfully.")


if __name__ == "__main__":
    test_c_adamw()