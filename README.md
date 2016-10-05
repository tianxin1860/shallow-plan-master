# shallow-plan
Word2Vec approach for completing plans with missing actions

### Commands

For training and testing

```bash
python train_and_test.py blocks t
```

For testing (using an already saved model)
```bash
python train_and_test.py blocks
```

### Outputs

```bash
$> python train_and_test.py blocks

=== Domain : blocks ===

==== FINAL STATISTICS ====

Total unknown actions: 19097; Total correct predictions: 9882
ACCURACY: 51.7463475939

=== Domain : depots ===

==== FINAL STATISTICS ====

Total unknown actions: 15028; Total correct predictions: 8192
ACCURACY: 54.511578387

=== Domain : driverlog ===

==== FINAL STATISTICS ====

Total unknown actions: 13614; Total correct predictions: 7041
ACCURACY: 51.7188188629

```
