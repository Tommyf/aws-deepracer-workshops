#!/usr/bin/env python

'''
Get details of RoboMaker jobs for DeepRacer

'''
import boto3
from operator import itemgetter
from json import loads
import ipywidgets as widgets

autoParams = {}

def get_robo_maker_jobs(jobsType='training'):
    rmclient = boto3.client('robomaker')
    # Get the list of RoboMaker simulation jobs that were used for DeepRacer
    deepRacerSimAppId = rmclient.list_simulation_applications()['simulationApplicationSummaries'][0]['name']
    response = rmclient.list_simulation_jobs(
        maxResults=100,
        filters=[
            {   
                'name': 'simulationApplicationName',
                'values': [
                    deepRacerSimAppId,
                ],
            }
        ]
    )
    rmjobs = response['simulationJobSummaries']
    allJobs={}
    allJobs['training'] = []
    allJobs['evaluation'] = []
    # Also get the summaries for each job and add that into the array
    for job in rmjobs:
        # Get & populate job summary
        job['summary'] = rmclient.describe_simulation_job(job=job['arn'])

        if job['summary']['simulationApplications'][0]['launchConfig']['launchFile'] == "distributed_training.launch":
            job['type'] = "training"
        elif job['summary']['simulationApplications'][0]['launchConfig']['launchFile'] == "evaluation.launch":
            job['type'] = "evaluation"

        job['id'] = job['arn'].partition("/")[-1]
        if 'lastStartedAt' in job['summary'].keys():
            job['startTime'] = job['summary']['lastStartedAt']
        else:
            job['startTime'] = None
        job['maxduration'] = job['summary']['maxJobDurationInSeconds']
        job['track'] = job['summary']['simulationApplications'][0]['launchConfig']['environmentVariables']['WORLD_NAME']

        if job['type'] == "training" and job['status'] != "Failed":
            # Get location of metadata (action space) and hyperparams
            job['metadataS3bucket'] = job['summary']['simulationApplications'][0]['launchConfig']['environmentVariables']['METRICS_S3_BUCKET']
            job['metadatafilekey'] = job['summary']['simulationApplications'][0]['launchConfig']['environmentVariables']['MODEL_METADATA_FILE_S3_KEY']
        
            job['hyperparamsS3bucket'] = job['summary']['simulationApplications'][0]['launchConfig']['environmentVariables']['SAGEMAKER_SHARED_S3_BUCKET']
            job['hyperparamsfilekey'] = "{}/ip/hyperparameters.json".format(job['summary']['simulationApplications'][0]['launchConfig']['environmentVariables']['SAGEMAKER_SHARED_S3_PREFIX'])

            # Download and ingest metadata and hyperparams
            s3 = boto3.resource('s3')
            job['actionspace'] = loads(s3.Object(job['metadataS3bucket'], job['metadatafilekey']).get()['Body'].read().decode('utf-8'))['action_space']
            job['hyperparams'] = loads(s3.Object(job['hyperparamsS3bucket'], job['hyperparamsfilekey']).get()['Body'].read().decode('utf-8'))
        
        # Create a description for the widget.
        job['desc'] = "{} - Type: {} - Track: {} - Duration: {}".format(job['id'],job['type'],job['track'],job['maxduration']/60)
        allJobs[job['type']].append(job)
    
    return allJobs[jobsType]

def get_job(jobsList,jobID):
    return list(filter(lambda job: job['id'] == jobID, jobsList))[0]

def display_job_selection_widget():
    
    output = widgets.Output(layout={'border': '1px solid black','width':'auto'})
    
    allJobs = get_robo_maker_jobs()
    with output:
        output.clear_output()    
    # Generate and display the dropdown list
    dropdownlist=list(map(itemgetter('desc','id'),allJobs))
    simSelectWidget = widgets.Dropdown(
        options=dropdownlist,
        value=dropdownlist[0][1],
        disabled=False,
        layout={'width':'auto'},
    )

    loadButtonText="Load Simulation Job Summary"
    print("Select RoboMaker Simulation Job ID and click on '{}' before continuing:".format(loadButtonText))
    display(simSelectWidget)
    
    button = widgets.Button(description=loadButtonText,layout={'width':'auto'})

    display(button, output)
    def on_button_clicked(b):
        global autoParams
        selectedJob = list(filter(lambda job: job['id'] == simSelectWidget.value, allJobs))[0]
        selectedJobID = selectedJob['id']
        track = selectedJob['track']

        with output:
            output.clear_output()
            print("Job ID: {}\t""Track: {}\n"
                  "Max Run Time: {}\t\tStart Time: {}\n"
                  "Status: {}\n".format(selectedJobID,
                                      track,
                                      selectedJob['maxduration']/60,
                                      selectedJob['startTime'],
                                      selectedJob['status']
                                      ))

            if 'actionspace' in selectedJob.keys():
                actionSpace = selectedJob['actionspace']
                autoParams['actionSpace'] = actionSpace
                print("Action Space:\nIndex")
                print("{: >10} {:>15.3} {: >10}".format("Index","Angle","Speed"))
                for action in actionSpace:
                    print("{: >10} {:>15f} {: >10}".format(action['index'],action['steering_angle'],action['speed']))
                print("\n")


            if 'hyperparams' in selectedJob.keys():
                hyperParameters = selectedJob['hyperparams']
                autoParams['hyperParameters'] = hyperParameters
                print("Hyperparameters:\nBatch Size:\t\t\t{}\n"
                    "Entropy:\t\t\t{}\n"
                    "Discount Factor:\t\t{}\n"
                    "Loss Type:\t\t\t{}\n"
                    "Learning Rate:\t\t\t{}\n"
                    "Episodes per iteration\t\t{}\n"
                    "No Epochs\t\t\t{}\n".format(hyperParameters['batch_size'],
                                            hyperParameters['beta_entropy'],
                                            hyperParameters['discount_factor'],
                                            hyperParameters['loss_type'],
                                            hyperParameters['lr'],
                                            hyperParameters['num_episodes_between_training'],
                                            hyperParameters['num_epochs'],
                                            ))
            
        autoParams['selectedJob'] = selectedJob
        autoParams['selectedJobID'] = selectedJobID
        autoParams['track'] = track
        
            
    button.on_click(on_button_clicked)

def get_auto_params():
    global autoParams
    return autoParams


