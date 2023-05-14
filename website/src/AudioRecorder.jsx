import { useState, useRef } from "react";
import axios from "axios";
import Dropdown from 'react-dropdown';
import Grid from '@mui/material/Grid';
import './DropDown.css';
import './AudioRecorder.css';

const mimeType = "audio/webm";

const modelOptions = [
	{ value: 'cnn', label: 'CNN' },
	{ value: 'svm', label: 'SVM' }
];
const defaultModelOption = modelOptions[0];

const taskOptions = [
	{ value: 'ser', label: 'Emotion Recognition' },
	{ value: 'yesno', label: 'YES / NO' }
];
const defaultTaskOption = taskOptions[0];

const AudioRecorder = () => {
	const [permission, setPermission] = useState(false);

	const [model, setModel] = useState(defaultModelOption.value);

	const [task, setTask] = useState(defaultTaskOption.value);

	const mediaRecorder = useRef(null);

	const [recordingStatus, setRecordingStatus] = useState("inactive");

	const [stream, setStream] = useState(null);

	const [audio, setAudio] = useState(null);

	const [audioBlob, setAudioBlob] = useState(null);

	const [prediction, setPrediction] = useState(null);

	const [audioChunks, setAudioChunks] = useState([]);

	const getMicrophonePermission = async () => {
		if ("MediaRecorder" in window) {
			try {
				const mediaStream = await navigator.mediaDevices.getUserMedia({
					audio: true,
					video: false,
				});
				setPermission(true);
				setStream(mediaStream);
			} catch (err) {
				alert(err.message);
			}
		} else {
			alert("The MediaRecorder API is not supported in your browser.");
		}
	};

	const startRecording = async () => {
		setRecordingStatus("recording");
		setPrediction(null);
		
		const media = new MediaRecorder(stream, { type: mimeType });

		mediaRecorder.current = media;

		mediaRecorder.current.start();

		let localAudioChunks = [];

		mediaRecorder.current.ondataavailable = (event) => {
			if (typeof event.data === "undefined") return;
			if (event.data.size === 0) return;
			localAudioChunks.push(event.data);
		};

		setAudioChunks(localAudioChunks);
	};

	const stopRecording = async () => {
		setRecordingStatus("inactive");
		mediaRecorder.current.stop();

		mediaRecorder.current.onstop = async () => {

			const audioBlob_ = new Blob(audioChunks, { type: mimeType });
			setAudioBlob(audioBlob_);

			const audioUrl = URL.createObjectURL(audioBlob_);
			setAudio(audioUrl);

			setAudioChunks([]);

			await runPredict(audioBlob_, model, task);
		};
	};

	const runPredict = async (blob, model, task) => {

		setPrediction(null);

		if (blob === null) {
			if (audioBlob === null) {
				return;
			}
			blob = audioBlob;
		}

		const configJson = JSON.stringify({"task": task, "model": model});
		const configBlob = new Blob([configJson], {
			type: 'application/json'
		});

		const formData = new FormData();
		formData.append("audio", blob);
		formData.append("config", configBlob);
		
		const response = await axios.post(
			`${import.meta.env.VITE_INFERENCE_API_URL}/predict`, 
			formData, 
			{
				headers: {
					'Content-Type': 'multipart/form-data',
					'Access-Control-Allow-Origin': '*'
				}
			}
		);

		setPrediction(response.data.label)
	};

	const onModelChange = async (option) => {
		setModel(option.value);
		await runPredict(audioBlob, option.value, task);
	};

	const onTaskChange = async (option) => {
		setTask(option.value);
		await runPredict(audioBlob, model, option.value);
	};

	return (
		<div>
			<main>
				<Grid container spacing={2} columns={54}>
					<Grid item xs={14}></Grid>
					<Grid item xs={8}>
						<h2>Task</h2>
					</Grid>
					<Grid item xs={18}>
	  					<Dropdown options={taskOptions} onChange={onTaskChange} value={defaultTaskOption} placeholder="Select an option" />
					</Grid>
					<Grid item xs={14}></Grid>
					<Grid item xs={14}></Grid>
					<Grid item xs={8}>
						<h2>Model</h2>
					</Grid>
					<Grid item xs={18}>
	  					<Dropdown options={modelOptions} onChange={onModelChange} value={defaultModelOption} placeholder="Select an option" />
					</Grid>
					<Grid item xs={14}></Grid>
				</Grid>
				<div className="audio-controls">
					{!permission ? (
						<button onClick={getMicrophonePermission} type="button">
							Get Microphone
						</button>
					) : null}
					{permission && recordingStatus === "inactive" ? (
						<button onClick={startRecording} type="button">
							Start Recording
						</button>
					) : null}
					{recordingStatus === "recording" ? (
						<button onClick={stopRecording} type="button">
							Stop Recording
						</button>
					) : null}
				</div>
				{audio ? (
					<div className="audio-player">
						<audio src={audio} controls></audio>
					</div>
				) : null}
				{prediction ? (
					<Grid container spacing={2} columns={12}>
						<Grid item xs={8}>
							<h1>Prediction:</h1>
						</Grid>
						<Grid item xs={4}>
							<h1>{prediction}</h1>
						</Grid>
					</Grid>
				) : null}
			</main>
		</div>
	);
};

export default AudioRecorder;
