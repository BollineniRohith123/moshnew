// Perfect Speech-to-Speech Implementation with WebRTC and AudioWorklet
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Perfect Unmute Voice Assistant...');

    // --- DOM Elements ---
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const clearBtn = document.getElementById('clear-btn');
    const statusElement = document.getElementById('connection-status');
    const conversationDiv = document.getElementById('conversation');
    const audioLevelElement = document.getElementById('audio-level');
    const responseTimeElement = document.getElementById('response-time');
    const vadStatusElement = document.getElementById('vad-status');
    const performanceElement = document.getElementById('performance-stats');

    // --- State Variables ---
    let ws;
    let audioContext;
    let audioWorkletNode;
    let mediaStream;
    let isRecording = false;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    let reconnectTimeout;
    
    // Performance tracking
    let performanceStats = {
        messagesProcessed: 0,
        averageLatency: 0,
        errorCount: 0
    };

    // --- WebRTC Configuration (for future enhancement) ---
    const rtcConfiguration = {
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' }
        ]
    };

    // --- Utility Functions ---
    function updateStatus(status, className = 'disconnected') {
        statusElement.textContent = status;
        statusElement.className = `status-value ${className}`;
        console.log(`[Status] ${status}`);
    }

    function updateVADStatus(isActive) {
        if (vadStatusElement) {
            vadStatusElement.textContent = isActive ? 'Speaking' : 'Listening';
            vadStatusElement.className = `status-value ${isActive ? 'speaking' : 'listening'}`;
        }
    }

    function updatePerformanceStats(stats) {
        if (performanceElement) {
            performanceElement.innerHTML = `
                <div>Processed: ${stats.messagesProcessed}</div>
                <div>Avg Latency: ${stats.averageLatency.toFixed(2)}s</div>
                <div>Errors: ${stats.errorCount}</div>
            `;
        }
    }

    function addMessage(text, sender, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        messageDiv.appendChild(contentDiv);
        
        // Add metadata if available
        if (metadata.processingTime) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            metaDiv.textContent = `Processing: ${metadata.processingTime.total.toFixed(2)}s`;
            messageDiv.appendChild(metaDiv);
        }
        
        conversationDiv.appendChild(messageDiv);
        conversationDiv.scrollTop = conversationDiv.scrollHeight;
        
        // Add animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        requestAnimationFrame(() => {
            messageDiv.style.transition = 'all 0.3s ease';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        });
    }

    // --- WebSocket Management ---
    function connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        
        updateStatus('Connecting...');
        console.log(`[WebSocket] Attempting to connect to ${wsUrl}`);
        
        try {
            ws = new WebSocket(wsUrl);
        } catch (error) {
            console.error('[WebSocket] Instantiation failed:', error);
            addMessage(`WebSocket Error: Could not create connection. Check browser compatibility.`, 'system');
            updateStatus('Connection Failed', 'error');
            scheduleReconnect();
            return;
        }

        ws.onopen = () => {
            updateStatus('Connected', 'connected');
            startBtn.disabled = false;
            reconnectAttempts = 0;
            addMessage('🎉 Connection established! You can now start talking.', 'system');
            
            // Send ping to keep connection alive
            setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, 30000);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (e) {
                console.error('[WebSocket] Error parsing message:', e);
            }
        };

        ws.onclose = (event) => {
            updateStatus('Disconnected', 'disconnected');
            startBtn.disabled = true;
            stopBtn.disabled = true;
            
            if (event.code !== 1000) { // If not a clean close
                console.warn(`[WebSocket] Connection closed unexpectedly (Code: ${event.code})`);
                addMessage(`Connection lost. Attempting to reconnect...`, 'system');
                scheduleReconnect();
            }
        };

        ws.onerror = (error) => {
            console.error('[WebSocket] Error:', error);
            updateStatus('Connection Error', 'error');
            addMessage('WebSocket connection error occurred.', 'system');
        };
    }

    function scheduleReconnect() {
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000); // Exponential backoff
            
            updateStatus(`Reconnecting in ${delay/1000}s... (${reconnectAttempts}/${maxReconnectAttempts})`, 'disconnected');
            
            reconnectTimeout = setTimeout(() => {
                connectWebSocket();
            }, delay);
        } else {
            updateStatus('Connection Failed', 'error');
            addMessage('❌ Maximum reconnection attempts reached. Please refresh the page.', 'system');
        }
    }

    function handleWebSocketMessage(data) {
        if (data.type === 'pong') {
            return; // Keep-alive response
        }

        if (data.error) {
            addMessage(`⚠️ ${data.error}`, 'system');
            performanceStats.errorCount++;
            updatePerformanceStats(performanceStats);
            return;
        }

        if (data.transcription) {
            addMessage(data.transcription, 'user');
        }

        if (data.response_text) {
            addMessage(data.response_text, 'assistant', {
                processingTime: data.processing_time
            });
            
            // Update performance stats
            performanceStats.messagesProcessed++;
            if (data.processing_time && data.processing_time.total) {
                performanceStats.averageLatency = 
                    (performanceStats.averageLatency * (performanceStats.messagesProcessed - 1) + 
                     data.processing_time.total) / performanceStats.messagesProcessed;
            }
            updatePerformanceStats(performanceStats);
            
            // Play audio response
            if (data.response_audio && data.response_audio.length > 0) {
                playAudio(data.response_audio);
            }
        }
    }

    // --- Audio Management ---
    async function initializeAudioContext() {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000,
                latencyHint: 'interactive'
            });
            
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            
            // Load AudioWorklet
            await audioContext.audioWorklet.addModule('/static/audio-worklet.js');
            
            return true;
        } catch (error) {
            console.error('[Audio] Failed to initialize AudioContext:', error);
            addMessage('❌ Audio initialization failed. Please check browser compatibility.', 'system');
            return false;
        }
    }

    async function startRecording() {
        if (isRecording) return;
        
        try {
            updateStatus('Initializing Audio...', 'connecting');
            
            // Initialize audio context if needed
            if (!audioContext) {
                const success = await initializeAudioContext();
                if (!success) return;
            }
            
            // Get user media with optimal settings
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    latency: 0.01 // 10ms latency for real-time feel
                }
            });
            
            // Create audio source and worklet
            const source = audioContext.createMediaStreamSource(mediaStream);
            audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');
            
            // Handle audio data from worklet
            audioWorkletNode.port.onmessage = (event) => {
                if (event.data.type === 'audioData') {
                    handleAudioData(event.data);
                }
            };
            
            // Connect audio graph
            source.connect(audioWorkletNode);
            audioWorkletNode.connect(audioContext.destination);
            
            // Start recording
            audioWorkletNode.port.postMessage({ command: 'start' });
            
            isRecording = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            startBtn.classList.add('recording');
            updateStatus('🎤 Recording...', 'recording');
            
            addMessage('🎤 Started listening...', 'system');
            
        } catch (error) {
            console.error('[Recording] Failed to start:', error);
            addMessage(`❌ Microphone Error: ${error.message}. Please allow microphone access and try again.`, 'system');
            updateStatus('Mic Error', 'error');
        }
    }

    function stopRecording() {
        if (!isRecording) return;
        
        try {
            isRecording = false;
            
            // Stop worklet
            if (audioWorkletNode) {
                audioWorkletNode.port.postMessage({ command: 'stop' });
                audioWorkletNode.disconnect();
                audioWorkletNode = null;
            }
            
            // Stop media stream
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startBtn.classList.remove('recording');
            updateStatus('Connected', 'connected');
            audioLevelElement.textContent = '0%';
            updateVADStatus(false);
            
            addMessage('🛑 Stopped listening.', 'system');
            
        } catch (error) {
            console.error('[Recording] Error stopping:', error);
        }
    }

    function handleAudioData(data) {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        
        // Update UI with audio level and VAD status
        const level = Math.round(data.energy * 100);
        audioLevelElement.textContent = `${level}%`;
        updateVADStatus(data.vadActive);
        
        // Send audio data to server
        try {
            ws.send(JSON.stringify({
                type: 'audio',
                audio: data.data
            }));
        } catch (error) {
            console.error('[Audio] Error sending data:', error);
        }
    }

    async function playAudio(audioArray) {
        if (!audioContext) return;
        
        try {
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            
            const audioBuffer = audioContext.createBuffer(1, audioArray.length, 24000);
            audioBuffer.getChannelData(0).set(audioArray);

            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            
            // Add some audio processing for better quality
            const gainNode = audioContext.createGain();
            gainNode.gain.value = 0.8;
            
            source.connect(gainNode);
            gainNode.connect(audioContext.destination);
            source.start();
            
        } catch (error) {
            console.error('[Audio] Playback failed:', error);
            addMessage('❌ Error playing audio response.', 'system');
        }
    }

    // --- Event Listeners ---
    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    
    clearBtn.addEventListener('click', () => {
        conversationDiv.innerHTML = '';
        performanceStats = { messagesProcessed: 0, averageLatency: 0, errorCount: 0 };
        updatePerformanceStats(performanceStats);
        addMessage('🗑️ Chat cleared.', 'system');
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (event) => {
        if (event.code === 'Space' && event.ctrlKey) {
            event.preventDefault();
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }
    });

    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && isRecording) {
            // Optionally pause recording when tab is hidden
            console.log('[Visibility] Tab hidden while recording');
        }
    });

    // Handle beforeunload
    window.addEventListener('beforeunload', () => {
        if (isRecording) {
            stopRecording();
        }
        if (ws) {
            ws.close();
        }
    });

    // --- Initialize Application ---
    addMessage('🚀 Welcome to Perfect Voice Assistant! Connecting to server...', 'system');
    connectWebSocket();
    
    // Periodic status check
    setInterval(async () => {
        try {
            const response = await fetch('/status');
            const status = await response.json();
            console.log('[Status Check]', status);
        } catch (error) {
            console.error('[Status Check] Failed:', error);
        }
    }, 60000); // Check every minute
});