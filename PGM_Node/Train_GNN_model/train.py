import tasks
import configs

prog_args = configs.arg_parse()

if prog_args.dataset is not None:
    if prog_args.dataset == "enron":
        print("Train enron dataset")
        
    elif prog_args.dataset == "ppi_essential":
        print("Train ppi_essential dataset")
    
    elif prog_args.dataset == "eucore":
        print("Train eucore dataset")
        training_function = "tasks.task_eucore"
        eval(training_function)(prog_args)
    
    elif prog_args.dataset == "bitcoinalpha":
        print("Train bitcoinalpha dataset")
        training_function = "tasks.task_bitcoinalpha"
        eval(training_function)(prog_args)
    
    elif prog_args.dataset == "bitcoinotc":
        print("Train bitcoinotc dataset")
        training_function = "tasks.task_bitcoinotc"
        eval(training_function)(prog_args)
    
    elif prog_args.dataset == "epinions":
        print("Train epinions dataset")
        training_function = "tasks.task_epinions"
        eval(training_function)(prog_args)
        
    elif prog_args.dataset == "amazon":
        print("Train amazon dataset")
                
    else:
        print("Training synthetic dataset")
        training_function = "tasks.task_syn"
        eval(training_function)(prog_args)
        