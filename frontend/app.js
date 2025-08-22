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
        this.alertsList = document.getElementById('alertsList');
        this.summaryLoading = document.getElementById('summaryLoading');
        this.summaryText = document.getElementById('summaryText');
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
        this.showNotification(`Loaded ${presetName} preset!`, 'success');
    }

    async startProcessing() {
        if (this.isProcessing) return;

        const config = {
            video_path: this.videoPath.value.trim(),
            mode: this.analysisMode.value,
            prompt: this.promptText.value.trim()
        };

        // Validation
        if (!config.video_path) {
            this.showNotification('Please enter a video file path', 'error');
            return;
        }

        if (!config.prompt) {
            this.showNotification('Please enter an analysis prompt', 'error');
            return;
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
                this.lastAlertCheck = 0; // Reset alert counter
                this.updateModeDisplay(config.mode);
                
                // Show loader for summary mode
                if (config.mode === 'summary') {
                    this.showSummaryLoader();
                }
                
                this.showNotification('Video analysis started successfully!', 'success');
                
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
            this.videoStream.src = `/video_feed?t=${Date.now()}`;
            this.videoStream.onload = () => {
                this.showVideoStream();
            };
            this.videoStream.onerror = () => {
                // Keep showing loader if video fails to load
                console.log('Video stream not ready yet, keeping loader...');
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
        switch (message.type) {
            case 'status_update':
                this.updateStats(message.data);
                break;
            case 'new_alert':
                this.addAlert(message.data);
                break;
            default:
                console.log('Unknown WebSocket message type:', message.type);
        }
    }

    updateModeDisplay(mode) {
        if (mode === 'alert') {
            this.alertsSection.style.display = 'block';
            this.summarySection.style.display = 'none';
        } else if (mode === 'summary') {
            this.alertsSection.style.display = 'none';
            this.summarySection.style.display = 'block';
        } else {
            this.alertsSection.style.display = 'none';
            this.summarySection.style.display = 'none';
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
}

// Global functions
function clearAlerts() {
    if (window.app) {
        window.app.clearAlerts();
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SoteiraApp();
});