#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.4),
    on Do 31 Okt 2024 17:28:40 WET
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.4'
expName = 'detection'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'eye_tracking': False,
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/home/crombie/code/QABVR_psychopy/detection.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('instruction_key') is None:
        # initialise instruction_key
        instruction_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction_key',
        )
    if deviceManager.getDevice('done_calibrating') is None:
        # initialise done_calibrating
        done_calibrating = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='done_calibrating',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "setup" ---
    # Run 'Begin Experiment' code from code
    from psychopy_visionscience.noise import NoiseStim
    
    import time, atexit
    from pupil_labs.realtime_api.simple import discover_one_device
    
    def send_event(device, name, clock_offset):
        client_time = time.time_ns()
        device_time = client_time - clock_offset
        device.send_event(
            name,
            event_timestamp_unix_ns=device_time
        )
        return client_time, device_time
        
    def stop_recording(device):
        device.recording_stop_and_save()
        print("Recording stopped")
        device.close()
     
     if expInfo['eye_tracking']:
        device = discover_one_device(max_search_duration_seconds=10)
        if device is None:
            print("No eye-tracking device found.")
            raise SystemExit(-1)
    
    # --- Initialize components for Routine "instruction" ---
    instruction_text = visual.TextStim(win=win, name='instruction_text',
        text='Welcome to the detection experiment.\n\nOn each trial, begin by fixing your gaze on the cross. After the cross disappears, two patches of noise will appear one after the other. One of the noise patches will contain a small oriented grating. Please indicate whether the grating was in the first (press left) or second (press right) patch of noise. Try to keep your gaze in the center of the display.\n\nPress space to continue to the eye tracker calibration.',
        font='Arial',
        pos=(0, 0), height=1.0, wrapWidth=30.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instruction_key = keyboard.Keyboard(deviceName='instruction_key')
    
    # --- Initialize components for Routine "calibration" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(2.5, 2.5),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    done_calibrating = keyboard.Keyboard(deviceName='done_calibrating')
    calibration_prompt = visual.TextStim(win=win, name='calibration_prompt',
        text="Press 'space' when calibration is complete.",
        font='Arial',
        pos=(0, -10), height=2.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "detection_trial" ---
    fixation_D1 = visual.ShapeStim(
        win=win, name='fixation_D1', vertices='cross',
        size=(1, 1),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    grating_D1 = visual.GratingStim(
        win=win, name='grating_D1',
        tex='sin', mask='gauss', anchor='center',
        ori=45.0, pos=[0,0], size=(1, 1), sf=6.0, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='add',
        texRes=128.0, interpolate=True, depth=-2.0)
    noise_D1 = visual.GratingStim(
        win=win, name='noise_D1',
        tex='sin', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5), sf=None, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=0.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-3.0)
    grating_D2 = visual.GratingStim(
        win=win, name='grating_D2',
        tex='sin', mask='gauss', anchor='center',
        ori=45.0, pos=[0,0], size=(1, 1), sf=6.0, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='add',
        texRes=128.0, interpolate=True, depth=-5.0)
    noise_D2 = visual.GratingStim(
        win=win, name='noise_D2',
        tex='sin', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5), sf=None, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=0.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-6.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "setup" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('setup.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    setupComponents = []
    for thisComponent in setupComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "setup" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in setupComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "setup" ---
    for thisComponent in setupComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('setup.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "setup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction.started', globalClock.getTime(format='float'))
    instruction_key.keys = []
    instruction_key.rt = []
    _instruction_key_allKeys = []
    # keep track of which components have finished
    instructionComponents = [instruction_text, instruction_key]
    for thisComponent in instructionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_text* updates
        
        # if instruction_text is starting this frame...
        if instruction_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_text.frameNStart = frameN  # exact frame index
            instruction_text.tStart = t  # local t and not account for scr refresh
            instruction_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_text.started')
            # update status
            instruction_text.status = STARTED
            instruction_text.setAutoDraw(True)
        
        # if instruction_text is active this frame...
        if instruction_text.status == STARTED:
            # update params
            pass
        
        # *instruction_key* updates
        waitOnFlip = False
        
        # if instruction_key is starting this frame...
        if instruction_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_key.frameNStart = frameN  # exact frame index
            instruction_key.tStart = t  # local t and not account for scr refresh
            instruction_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_key.started')
            # update status
            instruction_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction_key.status == STARTED and not waitOnFlip:
            theseKeys = instruction_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction_key_allKeys.extend(theseKeys)
            if len(_instruction_key_allKeys):
                instruction_key.keys = _instruction_key_allKeys[-1].name  # just the last key pressed
                instruction_key.rt = _instruction_key_allKeys[-1].rt
                instruction_key.duration = _instruction_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction" ---
    for thisComponent in instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruction.stopped', globalClock.getTime(format='float'))
    # check responses
    if instruction_key.keys in ['', [], None]:  # No response was made
        instruction_key.keys = None
    thisExp.addData('instruction_key.keys',instruction_key.keys)
    if instruction_key.keys != None:  # we had a response
        thisExp.addData('instruction_key.rt', instruction_key.rt)
        thisExp.addData('instruction_key.duration', instruction_key.duration)
    thisExp.nextEntry()
    # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "calibration" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('calibration.started', globalClock.getTime(format='float'))
    done_calibrating.keys = []
    done_calibrating.rt = []
    _done_calibrating_allKeys = []
    # keep track of which components have finished
    calibrationComponents = [polygon, done_calibrating, calibration_prompt]
    for thisComponent in calibrationComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *polygon* updates
        
        # if polygon is starting this frame...
        if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon.frameNStart = frameN  # exact frame index
            polygon.tStart = t  # local t and not account for scr refresh
            polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon.started')
            # update status
            polygon.status = STARTED
            polygon.setAutoDraw(True)
        
        # if polygon is active this frame...
        if polygon.status == STARTED:
            # update params
            pass
        
        # *done_calibrating* updates
        waitOnFlip = False
        
        # if done_calibrating is starting this frame...
        if done_calibrating.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            done_calibrating.frameNStart = frameN  # exact frame index
            done_calibrating.tStart = t  # local t and not account for scr refresh
            done_calibrating.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(done_calibrating, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'done_calibrating.started')
            # update status
            done_calibrating.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(done_calibrating.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(done_calibrating.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if done_calibrating.status == STARTED and not waitOnFlip:
            theseKeys = done_calibrating.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _done_calibrating_allKeys.extend(theseKeys)
            if len(_done_calibrating_allKeys):
                done_calibrating.keys = _done_calibrating_allKeys[-1].name  # just the last key pressed
                done_calibrating.rt = _done_calibrating_allKeys[-1].rt
                done_calibrating.duration = _done_calibrating_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *calibration_prompt* updates
        
        # if calibration_prompt is starting this frame...
        if calibration_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_prompt.frameNStart = frameN  # exact frame index
            calibration_prompt.tStart = t  # local t and not account for scr refresh
            calibration_prompt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_prompt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'calibration_prompt.started')
            # update status
            calibration_prompt.status = STARTED
            calibration_prompt.setAutoDraw(True)
        
        # if calibration_prompt is active this frame...
        if calibration_prompt.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in calibrationComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration" ---
    for thisComponent in calibrationComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('calibration.stopped', globalClock.getTime(format='float'))
    # check responses
    if done_calibrating.keys in ['', [], None]:  # No response was made
        done_calibrating.keys = None
    thisExp.addData('done_calibrating.keys',done_calibrating.keys)
    if done_calibrating.keys != None:  # we had a response
        thisExp.addData('done_calibrating.rt', done_calibrating.rt)
        thisExp.addData('done_calibrating.duration', done_calibrating.duration)
    # Run 'End Routine' code from start_recording
    if expInfo['eye_tracking']:
        estimate = device.estimate_time_offset()
        clock_offset_ns = round(estimate.time_offset_ms.mean * 1000000)
        recording_id = device.recording_start()
        print(f"Started recording with id {recording_id}")
        print("Waiting 5s...")
        time.sleep(5)
    thisExp.nextEntry()
    # the Routine "calibration" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    detection_trials = data.TrialHandler(nReps=5.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions_detection.xlsx'),
        seed=None, name='detection_trials')
    thisExp.addLoop(detection_trials)  # add the loop to the experiment
    thisDetection_trial = detection_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisDetection_trial.rgb)
    if thisDetection_trial != None:
        for paramName in thisDetection_trial:
            globals()[paramName] = thisDetection_trial[paramName]
    
    for thisDetection_trial in detection_trials:
        currentLoop = detection_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisDetection_trial.rgb)
        if thisDetection_trial != None:
            for paramName in thisDetection_trial:
                globals()[paramName] = thisDetection_trial[paramName]
        
        # --- Prepare to start Routine "detection_trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('detection_trial.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from send_event
        grating_angle = np.random.choice(
            np.arange(-np.pi, np.pi,  2 * np.pi / 32)
            )
        grating_x = grating_eccentricity * np.cos(grating_angle)
        grating_y = grating_eccentricity * np.sin(grating_angle)
        
        if expInfo['eye_tracking']:
            send_event(device, 'trial_start', clock_offset_ns)
        grating_D1.setOpacity(grating_in_D1)
        grating_D1.setContrast(grating_contrast)
        grating_D1.setPos((grating_x, grating_y))
        noise_D1.setContrast(noise_contrast)
        # Run 'Begin Routine' code from noise_code_D1
        noise_D1 = NoiseStim(
            win=win, name='pink_noise_1', units='height',
            noiseImage=None, mask='circle',
            ori=0.0, pos=(0, 0), size=(1, 1), sf=None,
            phase=0.0,
            color=[1,1,1], colorSpace='rgb', opacity=None, blendmode='add', contrast=1.0,
            texRes=256, filter=None,
            noiseType='Filtered', noiseElementSize=[0.0625], 
            noiseBaseSf=8.0, noiseBW=1.0,
            noiseBWO=30.0, noiseOri=0.0,
            noiseFractalPower=-1.25,noiseFilterLower=1.0,
            noiseFilterUpper=8.0, noiseFilterOrder=-2.0,
            noiseClip=3.0, imageComponent='Phase', interpolate=False, depth=0.0)
        noise_D1.buildNoise()
        grating_D2.setOpacity(grating_in_D2)
        grating_D2.setContrast(grating_contrast)
        grating_D2.setPos((grating_x, grating_y))
        noise_D2.setContrast(noise_contrast)
        # Run 'Begin Routine' code from noise_code_D2
        noise_D2 = NoiseStim(
            win=win, name='pink_noise_1', units='height',
            noiseImage=None, mask='circle',
            ori=0.0, pos=(0, 0), size=(1, 1), sf=None,
            phase=0.0,
            color=[1,1,1], colorSpace='rgb', opacity=None, blendmode='add', contrast=1.0,
            texRes=256, filter=None,
            noiseType='Filtered', noiseElementSize=[0.0625], 
            noiseBaseSf=8.0, noiseBW=1.0,
            noiseBWO=30.0, noiseOri=0.0,
            noiseFractalPower=-1.25,noiseFilterLower=1.0,
            noiseFilterUpper=8.0, noiseFilterOrder=-2.0,
            noiseClip=3.0, imageComponent='Phase', interpolate=False, depth=0.0)
        noise_D2.buildNoise()
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        detection_trialComponents = [fixation_D1, grating_D1, noise_D1, grating_D2, noise_D2, key_resp]
        for thisComponent in detection_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "detection_trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_D1* updates
            
            # if fixation_D1 is starting this frame...
            if fixation_D1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                fixation_D1.frameNStart = frameN  # exact frame index
                fixation_D1.tStart = t  # local t and not account for scr refresh
                fixation_D1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_D1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_D1.started')
                # update status
                fixation_D1.status = STARTED
                fixation_D1.setAutoDraw(True)
            
            # if fixation_D1 is active this frame...
            if fixation_D1.status == STARTED:
                # update params
                pass
            
            # if fixation_D1 is stopping this frame...
            if fixation_D1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_D1.tStartRefresh + 0.9-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_D1.tStop = t  # not accounting for scr refresh
                    fixation_D1.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_D1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_D1.stopped')
                    # update status
                    fixation_D1.status = FINISHED
                    fixation_D1.setAutoDraw(False)
            
            # *grating_D1* updates
            
            # if grating_D1 is starting this frame...
            if grating_D1.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                # keep track of start time/frame for later
                grating_D1.frameNStart = frameN  # exact frame index
                grating_D1.tStart = t  # local t and not account for scr refresh
                grating_D1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grating_D1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grating_D1.started')
                # update status
                grating_D1.status = STARTED
                grating_D1.setAutoDraw(True)
            
            # if grating_D1 is active this frame...
            if grating_D1.status == STARTED:
                # update params
                pass
            
            # if grating_D1 is stopping this frame...
            if grating_D1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grating_D1.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    grating_D1.tStop = t  # not accounting for scr refresh
                    grating_D1.tStopRefresh = tThisFlipGlobal  # on global time
                    grating_D1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grating_D1.stopped')
                    # update status
                    grating_D1.status = FINISHED
                    grating_D1.setAutoDraw(False)
            
            # *noise_D1* updates
            
            # if noise_D1 is starting this frame...
            if noise_D1.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                # keep track of start time/frame for later
                noise_D1.frameNStart = frameN  # exact frame index
                noise_D1.tStart = t  # local t and not account for scr refresh
                noise_D1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(noise_D1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'noise_D1.started')
                # update status
                noise_D1.status = STARTED
                noise_D1.setAutoDraw(True)
            
            # if noise_D1 is active this frame...
            if noise_D1.status == STARTED:
                # update params
                pass
            
            # if noise_D1 is stopping this frame...
            if noise_D1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > noise_D1.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    noise_D1.tStop = t  # not accounting for scr refresh
                    noise_D1.tStopRefresh = tThisFlipGlobal  # on global time
                    noise_D1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'noise_D1.stopped')
                    # update status
                    noise_D1.status = FINISHED
                    noise_D1.setAutoDraw(False)
            
            # *grating_D2* updates
            
            # if grating_D2 is starting this frame...
            if grating_D2.status == NOT_STARTED and tThisFlip >= 2.25-frameTolerance:
                # keep track of start time/frame for later
                grating_D2.frameNStart = frameN  # exact frame index
                grating_D2.tStart = t  # local t and not account for scr refresh
                grating_D2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grating_D2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grating_D2.started')
                # update status
                grating_D2.status = STARTED
                grating_D2.setAutoDraw(True)
            
            # if grating_D2 is active this frame...
            if grating_D2.status == STARTED:
                # update params
                pass
            
            # if grating_D2 is stopping this frame...
            if grating_D2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grating_D2.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    grating_D2.tStop = t  # not accounting for scr refresh
                    grating_D2.tStopRefresh = tThisFlipGlobal  # on global time
                    grating_D2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grating_D2.stopped')
                    # update status
                    grating_D2.status = FINISHED
                    grating_D2.setAutoDraw(False)
            
            # *noise_D2* updates
            
            # if noise_D2 is starting this frame...
            if noise_D2.status == NOT_STARTED and tThisFlip >= 2.25-frameTolerance:
                # keep track of start time/frame for later
                noise_D2.frameNStart = frameN  # exact frame index
                noise_D2.tStart = t  # local t and not account for scr refresh
                noise_D2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(noise_D2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'noise_D2.started')
                # update status
                noise_D2.status = STARTED
                noise_D2.setAutoDraw(True)
            
            # if noise_D2 is active this frame...
            if noise_D2.status == STARTED:
                # update params
                pass
            
            # if noise_D2 is stopping this frame...
            if noise_D2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > noise_D2.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    noise_D2.tStop = t  # not accounting for scr refresh
                    noise_D2.tStopRefresh = tThisFlipGlobal  # on global time
                    noise_D2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'noise_D2.stopped')
                    # update status
                    noise_D2.status = FINISHED
                    noise_D2.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # was this correct?
                    if (key_resp.keys == str('correct_key')) or (key_resp.keys == 'correct_key'):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in detection_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "detection_trial" ---
        for thisComponent in detection_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('detection_trial.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str('correct_key').lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for detection_trials (TrialHandler)
        detection_trials.addData('key_resp.keys',key_resp.keys)
        detection_trials.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            detection_trials.addData('key_resp.rt', key_resp.rt)
            detection_trials.addData('key_resp.duration', key_resp.duration)
        # the Routine "detection_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 5.0 repeats of 'detection_trials'
    
    # Run 'End Experiment' code from code
    if expInfo['eye_tracking']:
        stop_recording(device)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
