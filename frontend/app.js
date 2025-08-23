/**
 * Soteira Frontend Application
 * Real-time video analysis dashboard
 */

class SoteiraApp {
    constructor() {
        this.isProcessing = false;
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.alerts = [];
        this.lastAlertCheck = 0;
        
        // Text-to-Speech for accessibility
        this.ttsEnabled = false;
        this.speechSynthesis = window.speechSynthesis;
        this.currentVoice = null;
        this.speechQueue = [];
        this.isSpeaking = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
        this.startStatusPolling();
        this.loadVideoPresets();
        
        // Initialize with placeholder state
        this.showVideoPlaceholder();
    }

    initializeElements() {
        // Control elements
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.videoPreset = document.getElementById('videoPreset');
        this.videoPath = document.getElementById('videoPath');
        this.analysisMode = document.getElementById('analysisMode');
        this.promptText = document.getElementById('promptText');
        
        // Display elements
        this.videoStream = document.getElementById('videoStream');
        this.videoLoader = document.getElementById('videoLoader');
        this.videoPlaceholder = document.getElementById('videoPlaceholder');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.streamStatus = document.getElementById('streamStatus');
        
        // Mode-specific sections
        this.alertsSection = document.getElementById('alertsSection');
        this.summarySection = document.getElementById('summarySection');
        this.qaSection = document.getElementById('qaSection');
        this.alertsList = document.getElementById('alertsList');
        this.summaryLoading = document.getElementById('summaryLoading');
        this.summaryText = document.getElementById('summaryText');
        
        // Q&A elements
        this.questionInput = document.getElementById('questionInput');
        this.askBtn = document.getElementById('askBtn');
        this.qaHistory = document.getElementById('qaHistory');
        
        // TTS controls
        this.ttsToggle = document.getElementById('ttsToggle');
        this.speedModeToggle = document.getElementById('speedModeToggle');
        this.voiceSelect = document.getElementById('voiceSelect');
        this.speechRate = document.getElementById('speechRate');
        this.speechVolume = document.getElementById('speechVolume');
        this.testTtsBtn = document.getElementById('testTtsBtn');
        
        // Streaming display elements
        this.streamingDisplay = document.getElementById('streamingDisplay');
        this.streamingText = document.getElementById('streamingText');
        
        // Streaming mode state
        this.speedModeEnabled = false;
        this.streamingBuffer = "";
        this.lastTokenTime = 0;
        this.currentStreamingText = "";
        this.streamingActive = false;
        this.streamingHistory = [];  // Keep history of completed descriptions
    }

    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startProcessing());
        this.stopBtn.addEventListener('click', () => this.stopProcessing());

        this.videoPreset.addEventListener('change', (e) => {
            this.loadPreset(e.target.value);
        });

        this.analysisMode.addEventListener('change', (e) => {
            this.updateModeDisplay(e.target.value);
        });

        // Auto-refresh video stream
        this.videoStream.addEventListener('error', () => {
            console.log('Video stream error, retrying...');
            setTimeout(() => {
                if (this.isProcessing) {
                    this.updateVideoStream();
                }
            }, 1000);
        });

        // TTS event listeners
        if (this.ttsToggle) {
            this.ttsToggle.addEventListener('change', (e) => {
                this.ttsEnabled = e.target.checked;
                if (!this.ttsEnabled) {
                    this.stopSpeech();
                }
            });
        }

        if (this.speedModeToggle) {
            this.speedModeToggle.addEventListener('change', (e) => {
                this.speedModeEnabled = e.target.checked;
                console.log(`[SPEED] Speed mode ${this.speedModeEnabled ? 'enabled' : 'disabled'}`);
                
                // Clear streaming state when toggling
                this.streamingBuffer = "";
                this.currentStreamingText = "";
                this.streamingActive = false;
                
                // Show/hide streaming display based on mode and speed setting
                if (this.streamingDisplay) {
                    const showStreaming = (this.speedModeEnabled || this.analysisMode.value === 'realtime_description') && 
                                        this.analysisMode.value === 'realtime_description';
                    this.streamingDisplay.style.display = showStreaming ? 'block' : 'none';
                    
                    if (showStreaming && this.streamingText) {
                        this.streamingText.textContent = 'Waiting for description...';
                    }
                }
                
                // Auto-enable TTS when speed mode is enabled
                if (this.speedModeEnabled && this.ttsToggle) {
                    this.ttsToggle.checked = true;
                    this.ttsEnabled = true;
                }
            });
        }

        if (this.voiceSelect) {
            this.voiceSelect.addEventListener('change', (e) => {
                this.currentVoice = this.speechSynthesis.getVoices().find(v => v.name === e.target.value);
            });
        }

        if (this.testTtsBtn) {
            this.testTtsBtn.addEventListener('click', () => {
                this.ttsEnabled = true; // Temporarily enable for testing
                this.speak('Testing text to speech. You are now hearing the accessibility voice for real-time scene descriptions.');
                
                // Also test the alert system
                setTimeout(() => {
                    this.addAlert({
                        id: 'test',
                        timestamp: new Date().toISOString(),
                        message: 'Test accessibility alert - this should be spoken aloud',
                        confidence: 0.9
                    });
                }, 2000);
            });
        }

        // Initialize TTS voices
        this.initializeTTS();
    }

    async loadVideoPresets() {
        try {
            const response = await fetch('/video_presets');
            const presets = await response.json();
            this.videoPresets = presets;
            console.log('Loaded video presets:', presets);
        } catch (error) {
            console.error('Error loading presets:', error);
        }
    }

    loadPreset(presetName) {
        if (!presetName || !this.videoPresets || !this.videoPresets[presetName]) {
            return;
        }

        const preset = this.videoPresets[presetName];
        console.log('Loading preset:', presetName, preset);

        // Update form fields
        this.videoPath.value = preset.video_path;
        this.analysisMode.value = preset.mode;
        this.promptText.value = preset.prompt;

        this.updateModeDisplay(preset.mode);
        
        // Show specific notification for phone stream
        if (preset.video_path === 'phone') {
            this.showNotification(`Loaded ${presetName} preset! Make sure video_server.py is running for phone streaming.`, 'success');
        } else {
            this.showNotification(`Loaded ${presetName} preset!`, 'success');
        }
    }

    async startProcessing() {
        if (this.isProcessing) return;

        const config = {
            video_path: this.videoPath.value.trim(),
            mode: this.analysisMode.value,
            prompt: this.promptText.value.trim(),
            streaming_mode: this.speedModeEnabled || this.analysisMode.value === 'realtime_description'
        };

        // Validation
        if (!config.video_path) {
            this.showNotification('Please enter a video source (file path or "phone")', 'error');
            return;
        }

        if (!config.prompt) {
            this.showNotification('Please enter an analysis prompt', 'error');
            return;
        }
        
        // Special handling for phone streams
        if (config.video_path === 'phone') {
            this.showNotification('Starting phone stream analysis. Ensure video_server.py is running!', 'info');
        }

        try {
            this.setLoadingState(true);
            
            const response = await fetch('/start_processing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            const result = await response.json();

            if (response.ok) {
                this.isProcessing = true;
                this.updateControlsState();
                this.showVideoLoader(); // Show loading state
                this.clearAlerts();
                this.clearQAHistory(); // Clear previous Q&A when starting new video
                this.lastAlertCheck = 0; // Reset alert counter
                this.updateModeDisplay(config.mode);
                
                // Show loader for summary mode
                if (config.mode === 'summary') {
                    this.showSummaryLoader();
                }
                
                if (config.video_path === 'phone') {
                    this.showNotification('Phone stream analysis started! Connect your phone to the video server.', 'success');
                } else {
                    this.showNotification('Video analysis started successfully!', 'success');
                }
                
                // Start trying to load video stream after a short delay
                setTimeout(() => {
                    this.updateVideoStream();
                }, 1000);
            } else {
                throw new Error(result.detail || 'Failed to start processing');
            }
        } catch (error) {
            console.error('Error starting processing:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.setLoadingState(false);
        }
    }

    async stopProcessing() {
        if (!this.isProcessing) return;

        try {
            this.setLoadingState(true);
            
            const response = await fetch('/stop_processing', {
                method: 'POST'
            });

            const result = await response.json();

            if (response.ok) {
                this.isProcessing = false;
                this.updateControlsState();
                this.showNotification('Video analysis stopped', 'info');
                
                // Hide mode displays
                this.alertsSection.style.display = 'none';
                this.summarySection.style.display = 'none';
                if (this.qaSection) {
                    this.qaSection.style.display = 'none';
                }
                
                // Reset to placeholder state
                this.showVideoPlaceholder();
            } else {
                throw new Error(result.detail || 'Failed to stop processing');
            }
        } catch (error) {
            console.error('Error stopping processing:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.setLoadingState(false);
        }
    }

    showVideoLoader() {
        this.videoStream.style.display = 'none';
        this.videoPlaceholder.style.display = 'none';
        this.videoLoader.style.display = 'block';
    }

    showVideoPlaceholder() {
        this.videoStream.style.display = 'none';
        this.videoLoader.style.display = 'none';
        this.videoPlaceholder.style.display = 'block';
    }

    showVideoStream() {
        this.videoLoader.style.display = 'none';
        this.videoPlaceholder.style.display = 'none';
        this.videoStream.style.display = 'block';
    }

    updateVideoStream() {
        if (this.isProcessing) {
            // Add cache busting and faster refresh for phone streams
            const timestamp = Date.now();
            this.videoStream.src = `/video_feed?t=${timestamp}`;
            
            this.videoStream.onload = () => {
                this.showVideoStream();
                // Refresh video stream more frequently for phone streams
                if (this.videoPath.value === 'phone') {
                    setTimeout(() => {
                        if (this.isProcessing) {
                            this.updateVideoStream();
                        }
                    }, 100); // Refresh every 100ms for phone streams
                } else {
                    setTimeout(() => {
                        if (this.isProcessing) {
                            this.updateVideoStream();
                        }
                    }, 1000); // Normal refresh for other sources
                }
            };
            
            this.videoStream.onerror = () => {
                // Keep showing loader if video fails to load, retry faster
                console.log('Video stream not ready yet, keeping loader...');
                setTimeout(() => {
                    if (this.isProcessing) {
                        this.updateVideoStream();
                    }
                }, 500);
            };
        }
    }

    updateControlsState() {
        this.startBtn.disabled = this.isProcessing;
        this.stopBtn.disabled = !this.isProcessing;
        
        // Update status indicator
        if (this.isProcessing) {
            this.statusIndicator.className = 'status-indicator status-live';
            this.streamStatus.textContent = 'Live';
        } else {
            this.statusIndicator.className = 'status-indicator status-stopped';
            this.streamStatus.textContent = 'Stopped';
        }
    }

    setLoadingState(loading) {
        if (loading) {
            const loadingSpinner = '<span class="loading"></span>';
            if (this.isProcessing) {
                this.stopBtn.innerHTML = `${loadingSpinner} Stopping...`;
            } else {
                this.startBtn.innerHTML = `${loadingSpinner} Starting...`;
            }
        } else {
            this.startBtn.innerHTML = '🚀 Start Analysis';
            this.stopBtn.innerHTML = '⏹️ Stop';
        }
    }

    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
            };
            
            this.websocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.attemptReconnect();
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.attemptReconnect();
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            setTimeout(() => {
                this.connectWebSocket();
            }, 2000 * this.reconnectAttempts);
        } else {
            console.error('Max WebSocket reconnection attempts reached');
            this.showNotification('Connection lost. Please refresh the page.', 'error');
        }
    }

    handleWebSocketMessage(message) {
        console.log('[WS] Received message:', message.type, message.data);
        switch (message.type) {
            case 'status_update':
                this.updateStats(message.data);
                break;
            case 'new_alert':
                console.log('[WS] Processing new alert:', message.data);
                this.addAlert(message.data);
                break;
            case 'token_stream':
                console.log('[WS] 🎯 Processing streaming token:', message.data);
                this.handleStreamingToken(message.data);
                break;
            default:
                console.log('Unknown WebSocket message type:', message.type);
        }
    }

    updateModeDisplay(mode) {
        if (mode === 'alert' || mode === 'realtime_description') {
            this.alertsSection.style.display = 'block';
            this.summarySection.style.display = 'none';
            
            // Show streaming display for realtime_description mode
            if (mode === 'realtime_description' && this.streamingDisplay) {
                this.streamingDisplay.style.display = 'block';
            }
            
            // Auto-enable TTS for realtime_description mode
            if (mode === 'realtime_description' && this.ttsToggle) {
                this.ttsToggle.checked = true;
                this.ttsEnabled = true;
            }
            
            // Show Q&A for alert mode only (not realtime_description)
            if (mode === 'alert' && this.qaSection) {
                this.qaSection.style.display = 'block';
            } else if (this.qaSection) {
                this.qaSection.style.display = 'none';
            }
        } else if (mode === 'summary') {
            this.alertsSection.style.display = 'none';
            this.summarySection.style.display = 'block';
            if (this.streamingDisplay) {
                this.streamingDisplay.style.display = 'none';
            }
            
            // Show Q&A for summary mode
            if (this.qaSection) {
                this.qaSection.style.display = 'block';
            }
        } else {
            this.alertsSection.style.display = 'none';
            this.summarySection.style.display = 'none';
            if (this.streamingDisplay) {
                this.streamingDisplay.style.display = 'none';
            }
            if (this.qaSection) {
                this.qaSection.style.display = 'none';
            }
        }
    }

    showSummaryLoader() {
        this.summaryLoading.style.display = 'block';
        this.summaryText.style.display = 'none';
    }

    async showSummary(summaryText) {
        this.summaryLoading.style.display = 'none';
        this.summaryText.style.display = 'block';
        this.summaryText.querySelector('div').textContent = summaryText;
    }

    updateStats(status) {
        // Update processing state if it changed
        if (status.is_running !== this.isProcessing) {
            this.isProcessing = status.is_running;
            this.updateControlsState();
            
            if (this.isProcessing) {
                this.updateVideoStream();
            } else {
                // Video finished - check for summary
                if (this.analysisMode.value === 'summary') {
                    this.fetchAndShowSummary();
                }
            }
        }
    }

    async fetchAndShowSummary() {
        try {
            const response = await fetch('/summary');
            const data = await response.json();
            if (data.summary && data.summary.trim()) {
                await this.showSummary(data.summary);
            } else {
                await this.showSummary('No summary available yet.');
            }
        } catch (error) {
            console.error('Error fetching summary:', error);
            await this.showSummary('Error loading summary.');
        }
    }

    addAlert(alert) {
        this.alerts.unshift(alert); // Add to beginning
        this.renderAlerts();
        this.showNotification(`🚨 ${alert.message}`, 'alert');
        
        // Speak the alert if TTS is enabled
        this.speak(alert.message);
    }

    renderAlerts() {
        if (this.alerts.length === 0) {
            this.alertsList.innerHTML = '<div class="no-alerts">No alerts yet. Start video analysis to see real-time notifications.</div>';
            return;
        }

        const alertsHtml = this.alerts.map(alert => {
            const time = new Date(alert.timestamp).toLocaleTimeString();
            return `
                <div class="alert-item">
                    <div class="alert-time">${time}</div>
                    <div class="alert-message">${alert.message}</div>
                </div>
            `;
        }).join('');

        this.alertsList.innerHTML = alertsHtml;
    }

    clearAlerts() {
        this.alerts = [];
        this.renderAlerts();
    }

    async startStatusPolling() {
        // Status polling and alert checking
        setInterval(async () => {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                try {
                    const response = await fetch('/status');
                    const status = await response.json();
                    this.updateStats(status);
                } catch (error) {
                    console.error('Status polling error:', error);
                }
            }
            
            // Check for new alerts if in alert mode
            if (this.isProcessing && this.analysisMode.value === 'alert') {
                this.checkForNewAlerts();
            }
        }, 2000);
    }

    async checkForNewAlerts() {
        try {
            const response = await fetch('/alerts');
            const serverAlerts = await response.json();
            
            // Check for new alerts
            if (serverAlerts.length > this.lastAlertCheck) {
                const newAlerts = serverAlerts.slice(this.lastAlertCheck);
                newAlerts.forEach(alert => {
                    this.addAlert(alert);
                });
                this.lastAlertCheck = serverAlerts.length;
            }
        } catch (error) {
            console.error('Error checking alerts:', error);
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">${message}</div>
        `;

        // Add styles
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '12px 20px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: '500',
            fontSize: '14px',
            zIndex: '1000',
            maxWidth: '350px',
            opacity: '0',
            transform: 'translateX(100%)',
            transition: 'all 0.3s ease',
            boxShadow: '0 4px 12px rgba(0,0,0,0.2)'
        });

        // Set background color based on type
        const backgrounds = {
            success: 'linear-gradient(135deg, #00ff88, #00cc6a)',
            error: 'linear-gradient(135deg, #ff4444, #cc0000)',
            alert: 'linear-gradient(135deg, #ff6b35, #f7931e)',
            info: 'linear-gradient(135deg, #00d4ff, #0099cc)'
        };
        notification.style.background = backgrounds[type] || backgrounds.info;

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    // Text-to-Speech Methods
    initializeTTS() {
        if ('speechSynthesis' in window) {
            // Load voices
            const loadVoices = () => {
                const voices = this.speechSynthesis.getVoices();
                if (this.voiceSelect && voices.length > 0) {
                    this.voiceSelect.innerHTML = voices
                        .filter(voice => voice.lang.startsWith('en'))
                        .map(voice => `<option value="${voice.name}">${voice.name} (${voice.lang})</option>`)
                        .join('');
                    
                    // Select first English voice as default
                    this.currentVoice = voices.find(voice => voice.lang.startsWith('en'));
                }
            };

            loadVoices();
            this.speechSynthesis.addEventListener('voiceschanged', loadVoices);
        }
    }

    speak(text) {
        console.log('[TTS] speak() called with:', text);
        console.log('[TTS] ttsEnabled:', this.ttsEnabled);
        console.log('[TTS] speechSynthesis available:', 'speechSynthesis' in window);
        
        if (!this.ttsEnabled || !('speechSynthesis' in window) || !text.trim()) {
            console.log('[TTS] Skipping speech - conditions not met');
            return;
        }

        // Clean up the text for better speech
        const cleanText = text
            .replace(/\|/g, ',')
            .replace(/Items:/g, 'Items found:')
            .replace(/Confidence:/g, 'Confidence level:')
            .replace(/Scene update:/g, '')
            .replace(/Scene continues:/g, '')
            .replace(/Scene description:/g, '')
            .trim();

        console.log('[TTS] Speaking cleaned text:', cleanText);
        const utterance = new SpeechSynthesisUtterance(cleanText);
        
        if (this.currentVoice) {
            utterance.voice = this.currentVoice;
        }
        
        utterance.rate = this.speechRate ? this.speechRate.value : 0.9;
        utterance.volume = this.speechVolume ? this.speechVolume.value : 0.8;
        utterance.pitch = 1.0;

        utterance.onstart = () => {
            console.log('[TTS] Speech started');
            this.isSpeaking = true;
        };

        utterance.onend = () => {
            console.log('[TTS] Speech ended');
            this.isSpeaking = false;
            this.processNextSpeech();
        };

        utterance.onerror = (event) => {
            console.warn('[TTS] Speech synthesis error:', event.error);
            this.isSpeaking = false;
            this.processNextSpeech();
        };

        console.log('[TTS] Current voice:', this.currentVoice?.name);
        console.log('[TTS] Speech rate:', utterance.rate);
        console.log('[TTS] Speech volume:', utterance.volume);

        if (this.isSpeaking) {
            console.log('[TTS] Adding to queue');
            this.speechQueue.push(utterance);
        } else {
            console.log('[TTS] Speaking immediately');
            this.speechSynthesis.speak(utterance);
        }
    }

    processNextSpeech() {
        if (this.speechQueue.length > 0 && !this.isSpeaking) {
            const nextUtterance = this.speechQueue.shift();
            this.speechSynthesis.speak(nextUtterance);
        }
    }

    stopSpeech() {
        if ('speechSynthesis' in window) {
            this.speechSynthesis.cancel();
            this.speechQueue = [];
            this.isSpeaking = false;
        }
    }

    // Streaming Token Handling for Progressive Display and TTS
    handleStreamingToken(data) {
        console.log(`[STREAM] 🎪 Received token: "${data.token}", speedMode: ${this.speedModeEnabled}, mode: ${this.analysisMode.value}`);
        
        // Handle streaming tokens when in realtime_description mode or when speed mode is explicitly enabled
        if (!this.speedModeEnabled && this.analysisMode.value !== 'realtime_description') {
            console.log(`[STREAM] ❌ Skipping token - speed mode disabled and not in realtime_description mode`);
            return;
        }

        const token = data.token;
        this.streamingBuffer += token;
        this.currentStreamingText += token;
        this.lastTokenTime = Date.now();
        
        console.log(`[STREAM] ✅ Token processed. Buffer: "${this.streamingBuffer}", Total: "${this.currentStreamingText}"`)

        // Update visual display immediately
        this.updateStreamingDisplay();

        console.log(`[STREAM] Buffer: "${this.streamingBuffer}"`);

        // Check if we have a complete sentence or phrase to speak (only if TTS enabled)
        if (this.ttsEnabled && this.shouldSpeakBuffer()) {
            this.speakStreamingBuffer();
        }

        // Set timeout to speak remaining buffer if no more tokens arrive
        clearTimeout(this.streamTimeout);
        this.streamTimeout = setTimeout(() => {
            if (this.streamingBuffer.trim() && this.ttsEnabled) {
                console.log('[STREAM] Timeout reached, speaking remaining buffer');
                this.speakStreamingBuffer();
            }
            // Mark streaming as complete
            this.markStreamingComplete();
        }, 1500); // 1.5 second timeout
    }

    shouldSpeakBuffer() {
        const buffer = this.streamingBuffer.trim();
        
        // Speak if we have a complete sentence
        if (buffer.match(/[.!?]+\s*$/)) {
            return true;
        }
        
        // Speak if we have a substantial phrase (after comma)
        if (buffer.match(/,\s+\w+/)) {
            return true;
        }
        
        // Speak if buffer is getting long (>100 chars)
        if (buffer.length > 100) {
            return true;
        }
        
        return false;
    }

    speakStreamingBuffer() {
        if (!this.streamingBuffer.trim()) {
            return;
        }

        const textToSpeak = this.streamingBuffer.trim();
        console.log(`[STREAM] Speaking: "${textToSpeak}"`);
        
        // Speak the accumulated text
        this.speak(textToSpeak);
        
        // Clear the buffer
        this.streamingBuffer = "";
    }

    updateStreamingDisplay() {
        const streamingText = document.getElementById('streamingText');
        if (streamingText) {
            // Combine history with current streaming text
            let fullText = '';
            
            // Add previous descriptions with timestamps
            this.streamingHistory.forEach((description, index) => {
                const timestamp = new Date(description.timestamp).toLocaleTimeString();
                fullText += `[${timestamp}] ${description.text}\n\n`;
            });
            
            // Add current streaming text if any
            if (this.currentStreamingText) {
                fullText += `[Live] ${this.currentStreamingText}`;
            } else if (this.streamingHistory.length === 0) {
                fullText = 'Waiting for description...';
            }
            
            streamingText.textContent = fullText;
            
            // Auto-scroll to bottom to show latest content
            streamingText.scrollTop = streamingText.scrollHeight;
        }
    }

    markStreamingComplete() {
        const streamingIndicator = document.querySelector('.streaming-indicator');
        
        // Save current streaming text to history if it exists
        if (this.currentStreamingText.trim()) {
            this.streamingHistory.push({
                text: this.currentStreamingText.trim(),
                timestamp: Date.now()
            });
            
            // Keep only last 10 descriptions to prevent overflow
            if (this.streamingHistory.length > 10) {
                this.streamingHistory.shift();
            }
        }
        
        if (streamingIndicator) {
            streamingIndicator.style.animation = 'none';
            streamingIndicator.style.opacity = '0.3';
        }
        
        // Reset for next stream (but keep history)
        setTimeout(() => {
            if (streamingIndicator) {
                streamingIndicator.style.animation = 'pulse 1.5s infinite';
                streamingIndicator.style.opacity = '1';
            }
            this.currentStreamingText = '';
            this.updateStreamingDisplay(); // Refresh display to show history
        }, 1000); // Reduced delay
    }

    async askQuestion() {
        const question = this.questionInput.value.trim();
        if (!question) {
            this.showNotification('Please enter a question', 'warning');
            return;
        }

        // Check if we can ask questions
        if (!this.isProcessing && (!this.alerts || this.alerts.length === 0)) {
            this.showNotification('Process a video first before asking questions', 'warning');
            return;
        }

        try {
            // Show loading state
            this.askBtn.disabled = true;
            this.askBtn.textContent = 'Asking...';
            
            // Add loading indicator to Q&A history
            this.addQAItem(question, null, true);

            const response = await fetch('/ask_question', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to ask question');
            }

            const data = await response.json();
            
            // Remove loading indicator and add real Q&A
            this.removeLoadingQA();
            this.addQAItem(data.question, data.answer, false);
            
            // Clear input
            this.questionInput.value = '';
            
            this.showNotification('Question answered successfully!', 'success');

        } catch (error) {
            console.error('Error asking question:', error);
            this.removeLoadingQA();
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            // Reset button state
            this.askBtn.disabled = false;
            this.askBtn.textContent = 'Ask Question';
        }
    }

    addQAItem(question, answer, isLoading = false) {
        // Remove no-qa message if it exists
        const noQa = this.qaHistory.querySelector('.no-qa');
        if (noQa) {
            noQa.remove();
        }

        const qaItem = document.createElement('div');
        qaItem.className = 'qa-item';

        if (isLoading) {
            qaItem.innerHTML = `
                <div class="qa-question">${question}</div>
                <div class="qa-loading">
                    <div class="loading"></div>
                    <span>Analyzing video data and generating answer...</span>
                </div>
            `;
            qaItem.setAttribute('data-loading', 'true');
        } else {
            const timestamp = new Date().toLocaleTimeString();
            qaItem.innerHTML = `
                <div class="qa-question">Q: ${question}</div>
                <div class="qa-answer">${answer}</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem; text-align: right;">
                    ${timestamp}
                </div>
            `;
        }

        // Insert at the beginning (most recent first)
        this.qaHistory.insertBefore(qaItem, this.qaHistory.firstChild);
        
        // Scroll to the new item
        qaItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    removeLoadingQA() {
        const loadingItem = this.qaHistory.querySelector('.qa-item[data-loading="true"]');
        if (loadingItem) {
            loadingItem.remove();
        }
    }

    clearQAHistory() {
        if (this.qaHistory) {
            // Remove all Q&A items
            const qaItems = this.qaHistory.querySelectorAll('.qa-item');
            qaItems.forEach(item => item.remove());
            
            // Add back the no-qa message
            const noQaMsg = document.createElement('div');
            noQaMsg.className = 'no-qa';
            noQaMsg.style.textAlign = 'center';
            noQaMsg.style.color = '#999';
            noQaMsg.style.padding = '2rem';
            noQaMsg.style.fontStyle = 'italic';
            noQaMsg.textContent = 'Process a video first, then ask questions about what you observed.';
            this.qaHistory.appendChild(noQaMsg);
        }
    }
}

// Global functions
function clearAlerts() {
    if (window.app) {
        window.app.clearAlerts();
    }
}

function askQuestion() {
    if (window.app) {
        window.app.askQuestion();
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SoteiraApp();
});