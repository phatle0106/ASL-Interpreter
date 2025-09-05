import express from 'express';
import cors from 'cors';
import fetch from 'node-fetch';

// Fix for wrtc CommonJS module
import pkg from 'wrtc';
const { RTCPeerConnection, RTCSessionDescription, nonstandard } = pkg;

const app = express();

// Middleware
app.use(cors({
    origin: true, // Allow all origins for development
    credentials: true
}));
app.use(express.json({ limit: '10mb' }));

// Serve static files (for your HTML/CSS/JS)
app.use(express.static('.'));

// Configuration
const PORT = process.env.PORT || 8008;
const I3D_SERVICE_URL = process.env.I3D_SERVICE_URL || 'http://localhost:5000';
const T5_SERVICE_URL = process.env.T5_SERVICE_URL || 'http://localhost:5001';
const I3D_API_KEY = process.env.I3D_API_KEY || '';
const T5_API_KEY = process.env.T5_API_KEY || '';
const FRAME_QUEUE_SIZE = parseInt(process.env.FRAME_QUEUE_SIZE) || 5;
const PROCESSING_TIMEOUT = parseInt(process.env.PROCESSING_TIMEOUT) || 10000;
const ASL_PROCESSING_INTERVAL = parseInt(process.env.ASL_PROCESSING_INTERVAL) || 200;
const SENTENCE_GENERATION_THRESHOLD = parseInt(process.env.SENTENCE_GENERATION_THRESHOLD) || 5;

// Local development flag
const IS_LOCAL = process.env.NODE_ENV !== 'production';

console.log('Configuration:');
console.log(`   Port: ${PORT}`);
console.log(`   I3D Service URL: ${I3D_SERVICE_URL}`);
console.log(`   T5 Service URL: ${T5_SERVICE_URL}`);
console.log(`   Environment: ${process.env.NODE_ENV || 'development'}`);
console.log(`   Local development: ${IS_LOCAL}`);

// Store active connections and frame processing queues
const connections = new Map();
const processingQueues = new Map();
const userSessions = new Map(); // Track ASL sessions per user

// ASL-specific Frame processing queue class
class ASLFrameProcessor {
    constructor(connectionId, userId = null) {
        this.connectionId = connectionId;
        this.userId = userId || `conn_${connectionId}`;
        this.queue = [];
        this.processing = false;
        this.lastProcessedTime = Date.now();
        this.frameCounter = 0;
        this.lastRecognition = '';
        this.recognitionSequence = [];
        this.glossSequence = [];
        this.lastSentenceGeneration = Date.now();
        this.currentSentence = '';
        this.sentenceHistory = [];
    }
    
    async addFrame(frameData) {
        // Limit queue size for ASL (don't need as many frames buffered)
        if (this.queue.length >= FRAME_QUEUE_SIZE) {
            this.queue.shift(); // Remove oldest frame
        }
        
        this.queue.push({
            data: frameData,
            timestamp: Date.now()
        });
        
        if (!this.processing) {
            this.processQueue();
        }
    }
    
    async processQueue() {
        this.processing = true;
        
        while (this.queue.length > 0) {
            const frame = this.queue.shift();
            
            try {
                // Skip frame if too old (ASL needs fresher frames)
                if (Date.now() - frame.timestamp > 1000) {
                    continue;
                }
                
                this.frameCounter++;
                const result = await this.callI3DMicroservice(frame.data);
                
                // Process result and check for sentence generation
                const shouldGenerateSentence = await this.processI3DResult(result);
                
                // Send result back to client
                const connection = connections.get(this.connectionId);
                if (connection && connection.dataChannel && 
                    connection.dataChannel.readyState === 'open') {
                    
                    // Format result for client
                    const clientResult = this.formatASLResult(result);
                    connection.dataChannel.send(JSON.stringify(clientResult));
                    
                    // Generate sentence if conditions are met
                    if (shouldGenerateSentence) {
                        await this.generateSentence(connection);
                    }
                }
                
                this.lastProcessedTime = Date.now();
                
                // ASL processing interval (don't overwhelm the microservice)
                await new Promise(resolve => setTimeout(resolve, ASL_PROCESSING_INTERVAL));
                
            } catch (error) {
                console.error(`ASL processing error for connection ${this.connectionId}:`, error);
                
                // Send error message to client
                const connection = connections.get(this.connectionId);
                if (connection && connection.dataChannel && 
                    connection.dataChannel.readyState === 'open') {
                    connection.dataChannel.send(JSON.stringify({
                        type: 'asl_result',
                        text: 'ASL processing temporarily unavailable',
                        recognition: null,
                        confidence: 0.0,
                        error: true,
                        timestamp: new Date().toISOString()
                    }));
                }
            }
        }
        
        this.processing = false;
    }
    
    async callI3DMicroservice(base64FrameData) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), PROCESSING_TIMEOUT);
        
        try {
            const headers = {
                'Content-Type': 'application/json'
            };
            
            if (I3D_API_KEY) {
                headers['Authorization'] = `Bearer ${I3D_API_KEY}`;
            }

            // Prepare payload for I3D microservice
            const payload = {
                session_id: this.userId,
                frame: base64FrameData
            };

            console.log(`Sending frame to I3D microservice: ${I3D_SERVICE_URL}/process_frame`);

            const response = await fetch(`${I3D_SERVICE_URL}/process_frame`, {
                method: 'POST',
                headers,
                body: JSON.stringify(payload),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`I3D microservice returned ${response.status}: ${errorText}`);
            }

            const result = await response.json();
            return result;
            
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                console.error('I3D microservice request timeout');
                return { 
                    session_id: this.userId,
                    recognition: null,
                    current_gloss: '',
                    error: 'timeout',
                    timestamp: new Date().toISOString()
                };
            }
            
            console.error('I3D microservice call failed:', error);
            throw error;
        }
    }
    
    async callT5Microservice(glosses) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), PROCESSING_TIMEOUT);
        
        try {
            const headers = {
                'Content-Type': 'application/json'
            };
            
            if (T5_API_KEY) {
                headers['Authorization'] = `Bearer ${T5_API_KEY}`;
            }

            const payload = {
                session_id: this.userId,
                glosses: glosses
            };

            console.log(`Sending glosses to T5 microservice: ${T5_SERVICE_URL}/generate_sentence`);

            const response = await fetch(`${T5_SERVICE_URL}/generate_sentence`, {
                method: 'POST',
                headers,
                body: JSON.stringify(payload),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`T5 microservice returned ${response.status}: ${errorText}`);
            }

            const result = await response.json();
            return result;
            
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                console.error('T5 microservice request timeout');
                return { 
                    session_id: this.userId,
                    sentence: '',
                    error: 'timeout',
                    timestamp: new Date().toISOString()
                };
            }
            
            console.error('T5 microservice call failed:', error);
            throw error;
        }
    }
    
    async processI3DResult(i3dResponse) {
        let shouldGenerateSentence = false;
        
        // Track recognition sequence
        if (i3dResponse.recognition && i3dResponse.recognition !== this.lastRecognition) {
            this.recognitionSequence.push({
                word: i3dResponse.recognition,
                timestamp: new Date().toISOString()
            });
            
            // Add to gloss sequence for T5
            this.glossSequence.push(i3dResponse.recognition);
            
            this.lastRecognition = i3dResponse.recognition;
            
            // Keep only last 20 recognitions
            if (this.recognitionSequence.length > 20) {
                this.recognitionSequence = this.recognitionSequence.slice(-20);
            }
            
            if (this.glossSequence.length > 20) {
                this.glossSequence = this.glossSequence.slice(-20);
            }
            
            console.log(`Session ${this.userId} - Gloss sequence: ${this.glossSequence.join(' ')}`);
        }
        
        // Check if we should generate a sentence
        const timeSinceLastGeneration = Date.now() - this.lastSentenceGeneration;
        const hasEnoughGlosses = this.glossSequence.length >= SENTENCE_GENERATION_THRESHOLD;
        const hasTimePassed = timeSinceLastGeneration > 5000; // 5 seconds
        
        if (hasEnoughGlosses && hasTimePassed) {
            shouldGenerateSentence = true;
        }
        
        return shouldGenerateSentence;
    }
    
    async generateSentence(connection) {
        if (this.glossSequence.length === 0) {
            console.log(`No glosses available for sentence generation - session ${this.userId}`);
            return;
        }
        
        try {
            console.log(`Generating sentence for session ${this.userId} with glosses: ${this.glossSequence.join(' ')}`);
            
            const t5Result = await this.callT5Microservice([...this.glossSequence]);
            
            if (t5Result && t5Result.sentence && !t5Result.error) {
                this.currentSentence = t5Result.sentence;
                this.sentenceHistory.push({
                    sentence: t5Result.sentence,
                    glosses: [...this.glossSequence],
                    timestamp: new Date().toISOString()
                });
                
                // Keep only last 10 sentences
                if (this.sentenceHistory.length > 10) {
                    this.sentenceHistory = this.sentenceHistory.slice(-10);
                }
                
                this.lastSentenceGeneration = Date.now();
                
                console.log(`Generated sentence for session ${this.userId}: "${t5Result.sentence}"`);
                
                // Send sentence to client
                if (connection.dataChannel && connection.dataChannel.readyState === 'open') {
                    connection.dataChannel.send(JSON.stringify({
                        type: 'sentence_generated',
                        sentence: t5Result.sentence,
                        glosses: [...this.glossSequence],
                        confidence: t5Result.confidence || 0.8,
                        timestamp: new Date().toISOString(),
                        session_id: this.userId
                    }));
                }
                
                // Clear glosses after successful generation
                this.glossSequence = [];
                
            } else {
                console.error(`T5 sentence generation failed for session ${this.userId}:`, t5Result);
                
                // Send error to client
                if (connection.dataChannel && connection.dataChannel.readyState === 'open') {
                    connection.dataChannel.send(JSON.stringify({
                        type: 'sentence_error',
                        error: 'Failed to generate sentence',
                        glosses: [...this.glossSequence],
                        timestamp: new Date().toISOString()
                    }));
                }
            }
            
        } catch (error) {
            console.error(`Error generating sentence for session ${this.userId}:`, error);
            
            // Send error to client
            if (connection.dataChannel && connection.dataChannel.readyState === 'open') {
                connection.dataChannel.send(JSON.stringify({
                    type: 'sentence_error',
                    error: 'Sentence generation service unavailable',
                    glosses: [...this.glossSequence],
                    timestamp: new Date().toISOString()
                }));
            }
        }
    }
    
    formatASLResult(aslResponse) {
        return {
            type: 'asl_result',
            // New word recognized (only when new gesture detected)
            recognition: aslResponse.recognition || null,
            // Currently active gesture
            current_gloss: aslResponse.current_gloss || '',
            // Full sequence so far
            sentence: this.recognitionSequence.map(item => item.word).join(' '),
            // Current sentence from T5
            generated_sentence: this.currentSentence,
            // Processing status
            buffer_size: aslResponse.buffer_size || 0,
            frame_counter: aslResponse.frame_counter || this.frameCounter,
            session_id: aslResponse.session_id || this.userId,
            glosses_count: this.glossSequence.length,
            // Display text for UI
            text: this.formatDisplayText(aslResponse),
            confidence: this.calculateConfidence(aslResponse),
            timestamp: aslResponse.timestamp || new Date().toISOString(),
            error: aslResponse.error || false
        };
    }
    
    formatDisplayText(aslResponse) {
        const currentGloss = aslResponse.current_gloss || '';
        const glossSequence = this.glossSequence.join(' ');
        
        // Prioritize generated sentence if available
        if (this.currentSentence) {
            if (currentGloss) {
                return `"${this.currentSentence}" [${currentGloss}]`;
            }
            return `"${this.currentSentence}"`;
        } else if (currentGloss && glossSequence) {
            return `${glossSequence} [${currentGloss}]`;
        } else if (currentGloss) {
            return `[${currentGloss}]`;
        } else if (glossSequence) {
            return glossSequence;
        } else {
            return 'Ready for ASL recognition...';
        }
    }
    
    calculateConfidence(aslResponse) {
        // Simple confidence calculation based on buffer status
        const bufferSize = aslResponse.buffer_size || 0;
        const maxBuffer = 64; // From your ASL config
        
        if (this.currentSentence) {
            return 0.95; // Very high confidence for generated sentences
        } else if (aslResponse.recognition) {
            return 0.9; // High confidence for recognized words
        } else if (aslResponse.current_gloss) {
            return 0.7; // Medium confidence for active gesture
        } else if (bufferSize > maxBuffer * 0.5) {
            return 0.3; // Low confidence when building buffer
        } else {
            return 0.1; // Very low confidence
        }
    }
    
    async resetSession() {
        try {
            // Reset I3D session
            const headers = {
                'Content-Type': 'application/json'
            };
            
            if (I3D_API_KEY) {
                headers['Authorization'] = `Bearer ${I3D_API_KEY}`;
            }
            
            const i3dResponse = await fetch(`${I3D_SERVICE_URL}/reset_session`, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    session_id: this.userId
                })
            });
            
            // Reset T5 session
            const t5Headers = {
                'Content-Type': 'application/json'
            };
            
            if (T5_API_KEY) {
                t5Headers['Authorization'] = `Bearer ${T5_API_KEY}`;
            }
            
            const t5Response = await fetch(`${T5_SERVICE_URL}/reset_session`, {
                method: 'POST',
                headers: t5Headers,
                body: JSON.stringify({
                    session_id: this.userId
                })
            });
            
            const success = i3dResponse.ok && t5Response.ok;
            
            if (success) {
                // Reset local state
                this.recognitionSequence = [];
                this.glossSequence = [];
                this.lastRecognition = '';
                this.currentSentence = '';
                this.sentenceHistory = [];
                this.lastSentenceGeneration = Date.now();
                
                console.log(`Reset both I3D and T5 sessions for user: ${this.userId}`);
            }
            
            return success;
        } catch (error) {
            console.error('Failed to reset sessions:', error);
            return false;
        }
    }
    
    async requestSentenceGeneration() {
        try {
            if (this.glossSequence.length === 0) {
                return {
                    success: false,
                    message: 'No glosses available for sentence generation',
                    sentence: ''
                };
            }
            
            const t5Result = await this.callT5Microservice([...this.glossSequence]);
            
            if (t5Result && t5Result.sentence && !t5Result.error) {
                this.currentSentence = t5Result.sentence;
                this.sentenceHistory.push({
                    sentence: t5Result.sentence,
                    glosses: [...this.glossSequence],
                    timestamp: new Date().toISOString()
                });
                
                this.lastSentenceGeneration = Date.now();
                
                // Clear glosses after generation
                this.glossSequence = [];
                
                return {
                    success: true,
                    sentence: t5Result.sentence,
                    glosses: t5Result.glosses_input,
                    confidence: t5Result.confidence
                };
            } else {
                return {
                    success: false,
                    message: 'T5 service failed to generate sentence',
                    error: t5Result.error
                };
            }
            
        } catch (error) {
            console.error('Manual sentence generation error:', error);
            return {
                success: false,
                message: 'Sentence generation service unavailable',
                error: error.message
            };
        }
    }
}

// Health check for microservices
async function checkMicroserviceHealth(serviceUrl, serviceName) {
    try {
        const response = await fetch(`${serviceUrl}/health`, {
            method: 'GET',
            timeout: 5000
        });
        
        if (response.ok) {
            const health = await response.json();
            console.log(`${serviceName} Health:`, health);
            return health;
        }
        
        return false;
    } catch (error) {
        console.error(`${serviceName} health check failed:`, error);
        return false;
    }
}

// WebRTC offer endpoint
app.post('/offer', async (req, res) => {
    try {
        const { sdp, type, userId } = req.body;
        
        if (!sdp || type !== 'offer') {
            return res.status(400).json({ error: 'Invalid SDP offer' });
        }

        console.log('Received WebRTC offer from client', userId ? `(User: ${userId})` : '');

        // Create peer connection for this client
        const pc = new RTCPeerConnection({
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        });

        const connectionId = Date.now() + Math.random();
        const finalUserId = userId || `user_${connectionId}`;
        
        connections.set(connectionId, { 
            pc, 
            dataChannel: null, 
            userId: finalUserId 
        });

        // Create ASL frame processor for this connection
        const aslProcessor = new ASLFrameProcessor(connectionId, finalUserId);
        processingQueues.set(connectionId, aslProcessor);

        // Handle incoming data channel from client
        pc.ondatachannel = (event) => {
            const channel = event.channel;
            console.log('Received data channel:', channel.label);
            
            connections.get(connectionId).dataChannel = channel;
            
            channel.onopen = () => {
                console.log(`Data channel opened with client (User: ${finalUserId})`);
                // Send welcome message with ASL-specific info
                channel.send(JSON.stringify({
                    type: 'system',
                    text: 'Connected to ASL Recognition with T5 sentence generation! Start signing to see results.',
                    confidence: 1.0,
                    asl_ready: true,
                    t5_ready: true,
                    session_id: finalUserId
                }));
            };
            
            channel.onmessage = async (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'frame' && data.data) {
                        // Add frame to ASL processing queue
                        await aslProcessor.addFrame(data.data);
                    } else if (data.type === 'reset_asl') {
                        // Reset both ASL and T5 sessions
                        const resetSuccess = await aslProcessor.resetSession();
                        channel.send(JSON.stringify({
                            type: 'system',
                            text: resetSuccess ? 'ASL and T5 sessions reset successfully' : 'Failed to reset sessions',
                            confidence: 1.0,
                            session_reset: resetSuccess
                        }));
                    } else if (data.type === 'generate_sentence') {
                        // Manual sentence generation request
                        const result = await aslProcessor.requestSentenceGeneration();
                        if (result.success) {
                            channel.send(JSON.stringify({
                                type: 'sentence_generated',
                                sentence: result.sentence,
                                glosses: result.glosses,
                                confidence: result.confidence || 0.8,
                                manual_request: true,
                                timestamp: new Date().toISOString()
                            }));
                        } else {
                            channel.send(JSON.stringify({
                                type: 'sentence_error',
                                error: result.message,
                                manual_request: true,
                                timestamp: new Date().toISOString()
                            }));
                        }
                    } else if (data.type === 'get_sentence') {
                        // Send current recognized sentence
                        channel.send(JSON.stringify({
                            type: 'current_sentence',
                            sentence: aslProcessor.currentSentence,
                            glosses: aslProcessor.glossSequence,
                            sentence_history: aslProcessor.sentenceHistory,
                            confidence: 1.0
                        }));
                    }
                } catch (error) {
                    console.error('Error processing client message:', error);
                    if (channel.readyState === 'open') {
                        channel.send(JSON.stringify({
                            type: 'error',
                            text: 'Failed to process request',
                            confidence: 0.0,
                            error: true
                        }));
                    }
                }
            };
            
            channel.onclose = () => {
                console.log(`Data channel closed for user: ${finalUserId}`);
                connections.delete(connectionId);
                processingQueues.delete(connectionId);
            };
        };

        // Handle incoming video track
        pc.ontrack = (event) => {
            console.log(`Received video track from client (User: ${finalUserId})`);
        };

        // Handle connection state changes
        pc.oniceconnectionstatechange = () => {
            console.log(`ICE connection state for ${finalUserId}: ${pc.iceConnectionState}`);
            if (pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'disconnected') {
                connections.delete(connectionId);
                processingQueues.delete(connectionId);
                try { pc.close(); } catch (e) {}
            }
        };

        // Set remote description and create answer
        await pc.setRemoteDescription(new RTCSessionDescription({ type, sdp }));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);

        // Return answer to client
        res.json({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type,
            session_id: finalUserId
        });

        console.log(`WebRTC handshake completed for user: ${finalUserId}`);

    } catch (error) {
        console.error('WebRTC setup error:', error);
        res.status(500).json({ error: error.message });
    }
});

// ASL-specific endpoints
app.post('/asl/reset-session', async (req, res) => {
    try {
        const { userId, connectionId } = req.body;
        
        let processor = null;
        
        if (connectionId && processingQueues.has(connectionId)) {
            processor = processingQueues.get(connectionId);
        } else if (userId) {
            // Find processor by userId
            for (const [id, proc] of processingQueues) {
                if (proc.userId === userId) {
                    processor = proc;
                    break;
                }
            }
        }
        
        if (processor) {
            const success = await processor.resetSession();
            res.json({ success, message: success ? 'Both I3D and T5 sessions reset' : 'Reset failed' });
        } else {
            res.status(404).json({ error: 'Session not found' });
        }
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/asl/generate-sentence', async (req, res) => {
    try {
        const { userId, connectionId } = req.body;
        
        let processor = null;
        
        if (connectionId && processingQueues.has(connectionId)) {
            processor = processingQueues.get(connectionId);
        } else if (userId) {
            for (const [id, proc] of processingQueues) {
                if (proc.userId === userId) {
                    processor = proc;
                    break;
                }
            }
        }
        
        if (processor) {
            const result = await processor.requestSentenceGeneration();
            res.json(result);
        } else {
            res.status(404).json({ error: 'Session not found' });
        }
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/asl/sessions', (req, res) => {
    const sessions = [];
    for (const [connectionId, processor] of processingQueues) {
        sessions.push({
            connectionId,
            userId: processor.userId,
            frameCounter: processor.frameCounter,
            recognitionCount: processor.recognitionSequence.length,
            glossesCount: processor.glossSequence.length,
            currentSentence: processor.currentSentence,
            sentenceHistory: processor.sentenceHistory.length,
            lastProcessed: processor.lastProcessedTime,
            currentGlossSequence: processor.glossSequence.join(' ')
        });
    }
    
    res.json({
        activeSessions: sessions.length,
        sessions
    });
});

// Health check endpoint
app.get('/health', async (req, res) => {
    const i3dHealth = await checkMicroserviceHealth(I3D_SERVICE_URL, 'I3D Service');
    const t5Health = await checkMicroserviceHealth(T5_SERVICE_URL, 'T5 Service');
    
    res.json({ 
        status: 'healthy',
        connections: connections.size,
        processingQueues: processingQueues.size,
        microservices: {
            i3d: {
                url: I3D_SERVICE_URL,
                healthy: !!i3dHealth,
                details: i3dHealth
            },
            t5: {
                url: T5_SERVICE_URL,
                healthy: !!t5Health,
                details: t5Health
            }
        },
        config: {
            frameQueueSize: FRAME_QUEUE_SIZE,
            processingTimeout: PROCESSING_TIMEOUT,
            processingInterval: ASL_PROCESSING_INTERVAL,
            sentenceThreshold: SENTENCE_GENERATION_THRESHOLD
        },
        timestamp: new Date().toISOString()
    });
});

// Test endpoint for microservices
app.post('/test-pipeline', async (req, res) => {
    try {
        const { image, userId } = req.body;
        
        if (!image) {
            return res.status(400).json({ error: 'No image provided' });
        }
        
        const testUserId = userId || 'test_user';
        const processor = new ASLFrameProcessor('test', testUserId);
        
        // Test I3D
        const i3dResult = await processor.callI3DMicroservice(image);
        
        // Test T5 with sample glosses
        const testGlosses = ['hello', 'my', 'name', 'test'];
        const t5Result = await processor.callT5Microservice(testGlosses);
        
        res.json({
            i3d_result: i3dResult,
            t5_result: t5Result,
            test_glosses: testGlosses,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get microservice status
app.get('/asl/microservice-status', async (req, res) => {
    try {
        const i3dHealth = await checkMicroserviceHealth(I3D_SERVICE_URL, 'I3D Service');
        const t5Health = await checkMicroserviceHealth(T5_SERVICE_URL, 'T5 Service');
        
        res.json({
            i3d: {
                connected: !!i3dHealth,
                health: i3dHealth,
                url: I3D_SERVICE_URL
            },
            t5: {
                connected: !!t5Health,
                health: t5Health,
                url: T5_SERVICE_URL
            }
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log('Shutting down gateway...');
    
    // Reset all sessions
    const resetPromises = [];
    for (const [id, processor] of processingQueues) {
        resetPromises.push(processor.resetSession());
    }
    
    try {
        await Promise.all(resetPromises);
        console.log('All I3D and T5 sessions reset');
    } catch (error) {
        console.error('Error resetting sessions:', error);
    }
    
    // Close all peer connections
    for (const [id, conn] of connections) {
        try {
            if (conn.dataChannel) conn.dataChannel.close();
            if (conn.pc) conn.pc.close();
        } catch (e) {
            console.error('Error closing connection:', e);
        }
    }
    
    connections.clear();
    processingQueues.clear();
    process.exit(0);
});

// Initialize server
const startServer = async () => {
    // Check microservices on startup
    console.log('Checking microservice connections...');
    const i3dHealth = await checkMicroserviceHealth(I3D_SERVICE_URL, 'I3D Service');
    const t5Health = await checkMicroserviceHealth(T5_SERVICE_URL, 'T5 Service');
    
    if (i3dHealth) {
        console.log('I3D microservice is connected and healthy');
    } else {
        console.log('I3D microservice is not available - will retry during operation');
    }
    
    if (t5Health) {
        console.log('T5 microservice is connected and healthy');
    } else {
        console.log('T5 microservice is not available - will retry during operation');
    }
    
    app.listen(PORT, () => {
        console.log(`Gateway server running on port ${PORT}`);
        console.log(`I3D Service URL: ${I3D_SERVICE_URL}`);
        console.log(`T5 Service URL: ${T5_SERVICE_URL}`);
        console.log(`Active connections: ${connections.size}`);
        console.log(`Ready for ASL recognition with sentence generation!`);
    });
};

startServer();

export default app;