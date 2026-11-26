# Compatibility Matrix

## Version History

| Provider Version | Garak Version | Python | Key Dependencies | Notes |
|------------------|---------------|--------|------------------|-------|
| 0.5.1 | ==0.17.1+rhaiv.1 | >=3.12 | kfp>=2.14.6, eval-hub-sdk[adapter]~=1.0.4, sdg-hub~=0.9.4, boto3>=1.35.88 | Standard scans use `run.spec`. Legacy selection fields remain accepted. Garak does not replace `cas.expand_intent_tree` or `cas.trust_code_stubs`. |
| 0.5.0 | ==0.15.0+rhaiv.2 | >=3.12 | kfp>=2.14.6, eval-hub-sdk[adapter]>=0.1.7, boto3>=1.35.88 | Eval-hub only (Llama Stack removed) |
| 0.4.1 | ==0.14.1+rhaiv.7 | >=3.12 | kfp>=2.14.6, eval-hub-sdk[adapter]>=0.1.7, llama-stack>=0.5.0 | Dual-mode (Llama Stack + eval-hub) |
| 0.4.0 | ==0.14.0 | >=3.12 | kfp>=2.14.6, eval-hub-sdk[adapter]>=0.1.7, llama-stack>=0.5.0 | Added eval-hub adapter |
| 0.2.0 | ==0.14.0 | >=3.12 | kfp>=2.14.6, llama-stack>=0.5.0 | Last Llama Stack-only version |
| 0.1.3 | ==0.12.0 | >=3.12 | kfp>=2.14.6, llama-stack==0.2.18 | Thin deps, lazy kfp init |
| 0.1.2 | ==0.12.0 | >=3.12 | kfp>=2.14.6, llama-stack>=0.2.15 | Remote + inline |
| 0.1.1 | ==0.12.0 | >=3.12 | llama-stack>=0.2.15 | Initial release (inline only) |
