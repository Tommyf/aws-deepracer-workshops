#!/usr/bin/env python

'''
Get details of RoboMaker jobs for DeepRacer

'''
import boto3
from operator import itemgetter
from json import loads
import ipywidgets as widgets

autoParams = {}

def get_robo_maker_jobs():
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

    # Also get the summaries for each job and add that into the array
    for job in rmjobs:
        # Get & populate job summary
        job['summary'] = rmclient.describe_simulation_job(job=job['arn'])
        job['id'] = job['arn'].partition("/")[-1]
        if 'lastStartedAt' in job['summary'].keys():
            job['startTime'] = job['summary']['lastStartedAt']
        else:
            job['startTime'] = None
        job['maxduration'] = job['summary']['maxJobDurationInSeconds']
        job['track'] = job['summary']['simulationApplications'][0]['launchConfig']['environmentVariables']['WORLD_NAME']
    
        # Create a description for the widget.
        job['desc'] = "{} - Track: {} - Duration: {}".format(job['id'],job['track'],job['maxduration']/60)
    
    return rmjobs

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
        hyperParameters = get_hyper_parameters(stream_prefix=selectedJob['id'])

        with output:
            output.clear_output()
            print("Job ID: {}\t""Track: {}\n"
                  "Max Run Time: {}\t\tStart Time: {}\n"
                  "Status: {}".format(selectedJobID,
                                      track,
                                      selectedJob['maxduration']/60,
                                      selectedJob['startTime'],
                                      selectedJob['status']
                                      ))

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
        autoParams['hyperParameters'] = hyperParameters
            
    button.on_click(on_button_clicked)

def get_auto_params():
    global autoParams
    return autoParams

def get_hyper_parameters(log_group=None, stream_name=None, stream_prefix=None, start_time=None, end_time=None):
    client = boto3.client('logs')

    if start_time is None:
        start_time = 1451490400000  # 2018
    if end_time is None:
        end_time = 2000000000000  # 2033 #arbitrary future date
    if log_group is None:
        log_group = "/aws/robomaker/SimulationJobs"

    if stream_name is None and stream_prefix is None:
        print("Both stream name and prefix can't be None")
        return

    if stream_prefix:
        kwargs = {
            'logStreamNamePrefix': stream_prefix,
        }
    else:
        kwargs = {
            'logStreamNames': [stream_name],
        }

    kwargs['logGroupName'] = log_group
    kwargs['limit'] = 10000
    kwargs['startTime'] = start_time
    kwargs['endTime'] = end_time

    resp = client.filter_log_events(**kwargs)
    parsingHyperParams = False
    hyperParamsRaw=""
    for event in resp['events']:
        if parsingHyperParams:
            hyperParamsRaw += event['message']
            if event['message'] == "}":
                parsingHyperParams = False
        elif event['message'] == 'Using the following hyper-parameters':
            parsingHyperParams = True
    if hyperParamsRaw == "":
        hyperParamsRaw = "{}"
    return loads(hyperParamsRaw)

