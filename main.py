import os
from pathlib import Path

from configs.make_config import get_config
from optimizer.optimize import optimize
from person_box_detector.inference import detect_persons
from pipeline.lib.defs import Job
from pipeline.lib.defs import Pipeline
from post_processor.audio_adder import add_audio
from post_processor.overlay import overlay
from pre_processor.data_fetcher import fetch_data
from pre_processor.sampler import sample
from splitter.splitter import split

if os.getenv("ENVIRONMENT") == "test":
    DEBUG = True
else:
    DEBUG = False


def wrapper_function(pipeline: Pipeline) -> None:
    pipeline()


def clear_files():
    data_path = Path("./data/")

    for folder in data_path.iterdir():
        for stuff in folder.iterdir():
            if (
                stuff.name.endswith(".json")
                or stuff.name.endswith(".mp4")
                or stuff.name.endswith(".csv")
                or stuff.name.endswith("feather")
                or stuff.name.endswith("avi")
            ):
                stuff.unlink(missing_ok=True)
            if stuff.is_dir():
                for ele in stuff.iterdir():
                    ele.unlink(missing_ok=True)
                stuff.rmdir()


if __name__ == "__main__":

    """The basic procedure to compose a pipeline is done by doing the following steps:
        1. Read the config from the file under - './configs/*.hjson' - make a dict of Config objects
        2. Create the steps using the Job objects - each Job object requires a function and a Config object
        3. Put the above Job objects in any kind of iterable or collection (like List or Tuple) following a particular order.
        4. Make a Pipeline object using the list of Jobs created in Step - 3"""

    clear_files()

    # Step-1: get all details from config file
    dict_of_configs = get_config(path_to_config="./configs/config.hjson")

    # Step-2: declare jobs
    fetch_data_step = Job(func=fetch_data, conf=dict_of_configs["fetch_data"])
    sample_step = Job(func=sample, conf=dict_of_configs["sample"])
    detect_persons_step = Job(
        func=detect_persons, conf=dict_of_configs["detect_persons"]
    )
    split_step = Job(func=split, conf=dict_of_configs["split"])
    optimization_step = Job(func=optimize, conf=dict_of_configs["optimization"])
    overlay_step = Job(func=overlay, conf=dict_of_configs["overlay"])
    audio_adder_step = Job(func=add_audio, conf=dict_of_configs["audio_adder"])
    # upload_step = Job(func=upload_video, conf=collection_of_configs['upload'])

    # Step-3: the jobs below are put in a certain order for the pipeline
    list_of_jobs = (
        fetch_data_step,
        sample_step,
        detect_persons_step,
        split_step,
        optimization_step,
        overlay_step,
        audio_adder_step
        # upload_step
    )

    # Step-4: instantiate a pipeline object
    pipeline_1 = Pipeline(list_of_steps=list_of_jobs)

    # execute pipeline
    pipeline_1()

    # clear intermediate data created by the pipeline
    if not DEBUG:
        pipeline_1.clear()

    """below we see an example of how we can instantiate a pipeline with just a first step"""
    # pipeline_1 = Pipeline(start_step=fetch_data_step)
    # # now we could also add steps to the pipeline individually
    # pipeline_1.add_job(sample_step)
    # pipeline.add_job(detect_persons_step)
    # pipeline.add_job(split_step)
    # pipeline.add_job(optimization_step)
    # pipeline.add_job(overlay_step)
    # pipeline.add_job(upload_step)

    # # execute pipeline
    # pipeline_1()

    """Below we make another pipeline following the exact same steps described before.
    We do this to check for parallel execution - ideally a new pipeline means a new config"""
    # collection_of_configs_2 = get_config(path_to_config="./configs/config.json")

    # # declare jobs
    # fetch_data_step_2 = Job(func=fetch_data, conf=collection_of_configs_2['fetch_data'])
    # sample_step_2 = Job(func=sample, conf=collection_of_configs_2['sample'])
    # detect_persons_step_2 = Job(func=detect_persons, conf=collection_of_configs_2['detect_persons'])
    # split_step_2 = Job(func=split, conf=collection_of_configs_2['split'])

    # # the jobs below are put in a certain order for the pipeline
    # list_of_jobs_2 = [
    # 	fetch_data_step_2,
    #  	sample_step_2,
    #  	detect_persons_step_2,
    # ]
    # # declare another pipeline
    # pipeline_2 = Pipeline(list_of_steps=list_of_jobs_2)

    # # bundle the pipelines together
    # collection_of_pipelines = [pipeline_1, pipeline_2]

    # # run the pipelines in parallel
    # from multiprocessing import Pool
    # with Pool() as p:
    # 	res = p.map(wrapper_function, collection_of_pipelines)
