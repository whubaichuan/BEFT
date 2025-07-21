
TASKS=(SST2)
MODEL=${MODEL:-facebook/opt-1.3b}

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
EVAL=${EVAL:-1000}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi

DEV=""

EPOCH=5
BS=8
LR=1e-3
TASK_ARGS=""
GA=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    SST2)
        EPOCH=5
        ;;
    RTE)
        EPOCH=5
        ;;
    BoolQ)
        GA=$(expr $BS / 4) #only for opt-6.7B to use GA
        BS=4
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA"                
        ;;            
    CB) # It has <1000 training examples. 
        EPOCH=5
        LR=1e-2 
        GA=$(expr $BS / 1) #only for opt-6.7B to use GA
        BS=1
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA"          
        ;;
    WSC) # It has <1000 training examples. 
        LR=1e-2 
        EPOCH=5
        ;;
    Copa) # It has <1000 training examples.
        EPOCH=5
        LR=1e-4
        TASK_ARGS="--train_as_classification False"
        ;;
    MultiRC) # Can only fit real bsz = 2 on 80G A100
        GA=$(expr $BS / 2)
        BS=2
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA"
        ;;
    ReCoRD) # Can only fit real bsz = 2 on 80G A100
        EPOCH=5
        GA=$(expr $BS / 2)
        BS=2
        LR=1e-4
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    DROP) # Can only fit real bsz = 1 on 80G A100
        EPOCH=5
        MAX_LENGTH=1536 #only for opt-6.7B to use GA
        GA=$(expr $BS / 1)
        BS=1
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    SQuAD) # Can only fit real bsz = 2 on 80G A100
        EPOCH=5
        GA=$(expr $BS / 2)
        BS=2
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
esac

TAG=$MODE-$EPOCH-$BS-$LR-$SEED

echo "Running task: $TASK"
echo $TAG
echo "EPOCH: $EPOCH"
echo "BS: $BS"
echo "LR: $LR"
echo "SEED: $SEED"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

python run.py \
    --model_name $MODEL \
    ${MAX_LENGTH:+--max_length $MAX_LENGTH} \
    --task_name $TASK \
    --output_dir ./BEFT_results/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN ${DEV:+--num_dev $DEV} --num_eval $EVAL --logging_steps 10 \
    --trainer regular --fp16 \
    --learning_rate $LR --num_train_epochs $EPOCH --per_device_train_batch_size $BS \
    --save_strategy no \
    --train_as_classification \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"

