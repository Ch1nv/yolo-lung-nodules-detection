from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"D:\deeplearning\lung_nodules_od\results\yolo11n\weights\best.pt")

    metrics = model.val(verbose=True)

    names = model.names
    
    print("\n=== Overall metrics ===")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP75:      {metrics.box.all_ap[:, 5].mean():.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")

    print("\n=== Per-class metrics ===")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP75':<12} {'mAP50-95':<12}")

    for i, name in names.items():
        p, r, map50, map5095 = metrics.box.class_result(i)

        # all_ap shape 通常是 [num_classes, 10]
        # index 0 = AP50, index 5 = AP75
        map75 = metrics.box.all_ap[i][5]

        print(
            f"{name:<20} "
            f"{p:<12.4f} "
            f"{r:<12.4f} "
            f"{map50:<12.4f} "
            f"{map75:<12.4f} "
            f"{map5095:<12.4f}"
        )
########        
# 11l ##
########
#         Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
#          all         41         44      0.801       0.75      0.824      0.479
# nodule_type1         33         36      0.959       0.75      0.882      0.612
# nodule_type2          8          8      0.643       0.75      0.766      0.345

# === Overall metrics ===
# Precision:  0.8008
# Recall:     0.7500
# mAP50:      0.8237
# mAP75:      0.5526
# mAP50-95:   0.4786

# === Per-class metrics ===
# Class                Precision    Recall       mAP50        mAP75        mAP50-95    
# nodule_type1         0.9588       0.7500       0.8817       0.6858       0.6121      
# nodule_type2         0.6429       0.7500       0.7658       0.4195       0.3450 

########
# 11m ##
########
#         Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
#          all         41         44      0.776      0.659      0.798      0.469
# nodule_type1         33         36      0.936      0.819      0.931      0.613
# nodule_type2          8          8      0.616        0.5      0.666      0.325

# === Overall metrics ===
# Precision:  0.7764
# Recall:     0.6594
# mAP50:      0.7983
# mAP75:      0.4314
# mAP50-95:   0.4690

# === Per-class metrics ===
# Class                Precision    Recall       mAP50        mAP75        mAP50-95    
# nodule_type1         0.9365       0.8189       0.9308       0.7527       0.6131      
# nodule_type2         0.6164       0.5000       0.6657       0.1102       0.3249  

########
# 11s ##
########
#         Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
#          all         41         44      0.883      0.825      0.915      0.538
# nodule_type1         33         36      0.866      0.899      0.953       0.67
# nodule_type2          8          8      0.899       0.75      0.878      0.405

# === Overall metrics ===
# Precision:  0.8826
# Recall:     0.8245
# mAP50:      0.9154
# mAP75:      0.5916
# mAP50-95:   0.5375

# === Per-class metrics ===
# Class                Precision    Recall       mAP50        mAP75        mAP50-95    
# nodule_type1         0.8662       0.8990       0.9526       0.8449       0.6698      
# nodule_type2         0.8991       0.7500       0.8781       0.3382       0.4052   

########        
# 11n ##
########
#         Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
#          all         41         44      0.854      0.897      0.898      0.654
# nodule_type1         33         36      0.943      0.972      0.982      0.733
# nodule_type2          8          8      0.766      0.822      0.813      0.575

# === Overall metrics ===
# Precision:  0.8545
# Recall:     0.8970
# mAP50:      0.8979
# mAP75:      0.7323
# mAP50-95:   0.6540

# === Per-class metrics ===
# Class                Precision    Recall       mAP50        mAP75        mAP50-95    
# nodule_type1         0.9430       0.9722       0.9825       0.8957       0.7325      
# nodule_type2         0.7660       0.8218       0.8133       0.5689       0.5755 