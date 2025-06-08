import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Container,
  Paper,
  TextField,
  IconButton,
  Typography,
  CircularProgress,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Button,
  Fade,
  Divider,
} from '@mui/material';
import { Mic, MicOff, Send, Refresh } from '@mui/icons-material';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import axios from 'axios';

// Create a custom theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    secondary: {
      main: '#f50057',
      light: '#ff4081',
      dark: '#c51162',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
      color: '#1976d2',
    },
    h6: {
      fontWeight: 500,
      color: '#424242',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 3px 10px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
  },
});

interface Message {
  text: string;
  isUser: boolean;
  timestamp: Date;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState<string>('');
  const [isComplete, setIsComplete] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Welcome message as an array for formatting
  const welcomeLines = [
    "Welcome to the Diabetes Risk Assessment Chatbot! 👋",
    "",
    "I'll help you assess your diabetes risk by asking a few questions about your health metrics. You can answer either by typing or using voice input.",
    "",
    "The assessment includes questions about:",
    "• Number of pregnancies",
    "• Glucose levels",
    "• Blood pressure",
    "• Skin thickness",
    "• Insulin levels",
    "• BMI",
    "• Diabetes pedigree function",
    "• Age",
    "",
    "Your privacy is important - all data is processed locally and not stored.",
    "",
    "Ready to begin? Click 'Start Assessment' to proceed!"
  ];

  // Configure axios to include credentials
  axios.defaults.withCredentials = true;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getCurrentQuestion = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/current-question');
      if (response.data.status === 'success') {
        if (response.data.is_complete) {
          setIsComplete(true);
          setCurrentQuestion('');
        } else {
          setCurrentQuestion(response.data.question);
          addMessage(response.data.question, false);
        }
      }
    } catch (error) {
      console.error('Error getting current question:', error);
      addMessage('Error connecting to the server. Please try again.', false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await processAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      addMessage('Error accessing microphone. Please check your permissions.', false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('audio', audioBlob);

    try {
      const response = await axios.post('http://localhost:5000/api/process-voice', formData);
      handleResponse(response.data);
    } catch (error) {
      console.error('Error processing audio:', error);
      addMessage('Sorry, there was an error processing your voice input. Please try again.', false);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleResponse = (data: any) => {
    if (data.status === 'success') {
      addMessage(data.message, false);
      if (data.is_complete) {
        getPrediction();
      } else {
        setCurrentQuestion(data.next_feature);
      }
    } else {
      addMessage(data.message, false);
    }
  };

  const getPrediction = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/predict');
      if (response.data.status === 'success') {
        addMessage(response.data.message, false);
        setIsComplete(true);
        setCurrentQuestion('');
      } else {
        addMessage(response.data.message, false);
      }
    } catch (error) {
      console.error('Error getting prediction:', error);
      addMessage('Error getting prediction. Please try again.', false);
    }
  };

  const addMessage = (text: string, isUser: boolean) => {
    setMessages(prev => [...prev, { text, isUser, timestamp: new Date() }]);
  };

  const handleSend = async () => {
    if (!inputText.trim() || isProcessing) return;

    addMessage(inputText, true);
    setInputText('');
    setIsProcessing(true);

    try {
      const response = await axios.post('http://localhost:5000/api/process-text', {
        text: inputText
      });
      handleResponse(response.data);
    } catch (error) {
      console.error('Error processing text:', error);
      addMessage('Sorry, there was an error processing your input. Please try again.', false);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = async () => {
    try {
      await axios.post('http://localhost:5000/api/reset');
      setMessages([]);
      setIsComplete(false);
      getCurrentQuestion();
    } catch (error) {
      console.error('Error resetting session:', error);
      addMessage('Error resetting the session. Please try again.', false);
    }
  };

  const handleStartAssessment = () => {
    setShowWelcome(false);
    setMessages([]);
    getCurrentQuestion();
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #e3f2fd 0%, #f5f5f5 100%)',
          py: 4,
        }}
      >
        <Container maxWidth="md">
          <Paper
            elevation={3}
            sx={{
              height: '80vh',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              position: 'relative',
            }}
          >
            {/* Header */}
            <Box
              sx={{
                p: 2,
                bgcolor: 'primary.main',
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <MedicalServicesIcon sx={{ fontSize: 32 }} />
              <Typography variant="h5" component="h1" sx={{ fontWeight: 600 }}>
                Diabetes Risk Assessment
              </Typography>
            </Box>

            {/* Messages Area */}
            <Box
              sx={{
                flex: 1,
                overflow: 'auto',
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                gap: 2,
                bgcolor: '#fafafa',
              }}
            >
              {messages.map((message, index) => (
                <Fade in={true} key={index}>
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: message.isUser ? 'flex-end' : 'flex-start',
                      mb: 1,
                    }}
                  >
                    <Paper
                      elevation={1}
                      sx={{
                        p: 2,
                        maxWidth: '80%',
                        bgcolor: message.isUser ? 'primary.light' : 'white',
                        color: message.isUser ? 'white' : 'text.primary',
                        borderRadius: 2,
                        whiteSpace: 'pre-line',
                        borderTopRightRadius: message.isUser ? 2 : 12,
                        borderTopLeftRadius: message.isUser ? 12 : 2,
                        borderBottomRightRadius: message.isUser ? 12 : 2,
                        borderBottomLeftRadius: message.isUser ? 2 : 12,
                        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                        border: message.isUser ? 'none' : '1px solid rgba(0,0,0,0.1)',
                      }}
                    >
                      <Typography 
                        variant="body1"
                        sx={{
                          textAlign: message.isUser ? 'right' : 'left',
                        }}
                      >
                        {message.text}
                      </Typography>
                      <Typography
                        variant="caption"
                        sx={{
                          display: 'block',
                          mt: 1,
                          opacity: 0.7,
                          textAlign: message.isUser ? 'right' : 'left',
                        }}
                      >
                        {message.timestamp.toLocaleTimeString()}
                      </Typography>
                    </Paper>
                  </Box>
                </Fade>
              ))}
              {isLoading && (
                <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                  <CircularProgress size={24} />
                </Box>
              )}
              <div ref={messagesEndRef} />
            </Box>

            {/* Input Area */}
            {!showWelcome && (
              <Box
                sx={{
                  p: 2,
                  bgcolor: 'background.paper',
                  borderTop: '1px solid',
                  borderColor: 'divider',
                }}
              >
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    fullWidth
                    variant="outlined"
                    placeholder={currentQuestion || "Type your answer..."}
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSend();
                      }
                    }}
                    disabled={isComplete}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        borderRadius: 2,
                      },
                    }}
                  />
                  <IconButton
                    color={isRecording ? 'secondary' : 'primary'}
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isComplete}
                    sx={{
                      bgcolor: isRecording ? 'secondary.light' : 'primary.light',
                      color: 'white',
                      '&:hover': {
                        bgcolor: isRecording ? 'secondary.main' : 'primary.main',
                      },
                    }}
                  >
                    {isRecording ? <MicOff /> : <Mic />}
                  </IconButton>
                  <Button
                    variant="contained"
                    onClick={handleSend}
                    disabled={!inputText.trim() || isComplete}
                    endIcon={<Send />}
                    sx={{ minWidth: 100 }}
                  >
                    Send
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={handleReset}
                    startIcon={<Refresh />}
                    sx={{ minWidth: 100 }}
                  >
                    Reset
                  </Button>
                </Box>
              </Box>
            )}

            {/* Welcome Screen */}
            {showWelcome && (
              <Box
                sx={{
                  p: 4,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  gap: 3,
                }}
              >
                <Paper
                  elevation={2}
                  sx={{
                    maxWidth: 500,
                    width: '100%',
                    p: 3,
                    mb: 3,
                    bgcolor: '#f5faff',
                    borderRadius: 3,
                    maxHeight: 320,
                    overflowY: 'auto',
                  }}
                >
                  {welcomeLines.map((line, idx) =>
                    line.startsWith('•') ? (
                      <Typography key={idx} variant="body1" sx={{ ml: 2 }}>
                        {line}
                      </Typography>
                    ) : (
                      <Typography key={idx} variant={idx === 0 ? 'h6' : 'body1'} sx={{ fontWeight: idx === 0 ? 600 : 400, mb: line === '' ? 1 : 0 }}>
                        {line}
                      </Typography>
                    )
                  )}
                </Paper>
                <Button
                  variant="contained"
                  size="large"
                  onClick={handleStartAssessment}
                  startIcon={<MedicalServicesIcon />}
                  sx={{
                    py: 1.5,
                    px: 4,
                    fontSize: '1.1rem',
                  }}
                >
                  Start Assessment
                </Button>
              </Box>
            )}
          </Paper>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
