import re

with open('training/gemma4_reranker/trainer.py', 'r') as f:
    content = f.read()

# 1. Update compute_loss definition
old_def = r'def compute_loss\(\n    model,\n    inputs: dict,\n    yes_id: int,\n    no_id: int,\n    lm_head: torch\.nn\.Linear,\n    kd_alpha: float = 0\.0,\n    kd_temperature: float = 2\.0,\n\)'
new_def = 'def compute_loss(\n    model,\n    inputs: dict,\n    ctx: LossContext,\n)'
content = re.sub(old_def, new_def, content)

# 2. Update compute_loss docstring
old_doc = r'Args:\n        model: Gemma4ForConditionalGeneration \(PEFT wrapped\)\.\n        inputs: Batch dict with input_ids, attention_mask, labels\.\n        yes_id: Token ID for "yes"\.\n        no_id: Token ID for "no"\.\n        lm_head: lm_head Linear module for logit computation\.\n        kd_alpha: Knowledge distillation blend weight \(0 = no KD\)\.\n        kd_temperature: KD softmax temperature\.'
new_doc = 'Args:\n        model: Gemma4ForConditionalGeneration (PEFT wrapped).\n        inputs: Batch dict with input_ids, attention_mask, labels.\n        ctx: LossContext containing yes_id, no_id, lm_head, and KD settings.'
content = re.sub(old_doc, new_doc, content)

# 3. Update compute_loss body
content = content.replace('logits = F.linear(last_h, lm_head.weight)', 'logits = F.linear(last_h, ctx.lm_head.weight)')
content = content.replace('yes_logit = logits[:, yes_id]', 'yes_logit = logits[:, ctx.yes_id]')
content = content.replace('no_logit = logits[:, no_id]', 'no_logit = logits[:, ctx.no_id]')
content = content.replace('if kd_alpha > 0 and "teacher_scores" in inputs:', 'if ctx.kd_alpha > 0 and "teacher_scores" in inputs:')
content = content.replace('teacher_probs = F.softmax(teacher_logits / kd_temperature, dim=-1)', 'teacher_probs = F.softmax(teacher_logits / ctx.kd_temperature, dim=-1)')
content = content.replace('student_log_probs = F.log_softmax(logits_yes_no / kd_temperature, dim=-1)', 'student_log_probs = F.log_softmax(logits_yes_no / ctx.kd_temperature, dim=-1)')
content = content.replace('loss = (1 - kd_alpha) * loss_ce + kd_alpha * loss_kd * (kd_temperature**2)', 'loss = (1 - ctx.kd_alpha) * loss_ce + ctx.kd_alpha * loss_kd * (ctx.kd_temperature**2)')

# 4. Update caller in train_one_epoch
# From:
#             loss, metrics = compute_loss(
#                 model,
#                 batch,
#                 yes_id=yes_id,
#                 no_id=no_id,
#                 lm_head=lm_head,
#                 kd_alpha=config.kd_alpha,
#                 kd_temperature=config.kd_temperature,
#             )
# To:
#             loss, metrics = compute_loss(
#                 model,
#                 batch,
#                 ctx=LossContext(
#                     yes_id=yes_id,
#                     no_id=no_id,
#                     lm_head=lm_head,
#                     kd_alpha=config.kd_alpha,
#                     kd_temperature=config.kd_temperature,
#                 ),
#             )

old_call_train = r'loss, metrics = compute_loss\(\n                model,\n                batch,\n                yes_id=yes_id,\n                no_id=no_id,\n                lm_head=lm_head,\n                kd_alpha=config\.kd_alpha,\n                kd_temperature=config\.kd_temperature,\n            \)'
new_call_train = 'loss, metrics = compute_loss(\n                model,\n                batch,\n                ctx=LossContext(\n                    yes_id=yes_id,\n                    no_id=no_id,\n                    lm_head=lm_head,\n                    kd_alpha=config.kd_alpha,\n                    kd_temperature=config.kd_temperature,\n                ),\n            )'
content = re.sub(old_call_train, new_call_train, content)

# 5. Update caller in evaluate
# From:
#             loss, _ = compute_loss(model, batch, yes_id=yes_id, no_id=no_id, lm_head=lm_head)
# To:
#             loss, _ = compute_loss(
#                 model,
#                 batch,
#                 ctx=LossContext(yes_id=yes_id, no_id=no_id, lm_head=lm_head),
#             )
old_call_eval = r'loss, _ = compute_loss\(model, batch, yes_id=yes_id, no_id=no_id, lm_head=lm_head\)'
new_call_eval = 'loss, _ = compute_loss(\n                model,\n                batch,\n                ctx=LossContext(yes_id=yes_id, no_id=no_id, lm_head=lm_head),\n            )'
content = re.sub(old_call_eval, new_call_eval, content)

with open('training/gemma4_reranker/trainer.py', 'w') as f:
    f.write(content)
