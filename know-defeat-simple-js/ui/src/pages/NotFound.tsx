import { Box, Button, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import MainLayout from '../components/layout/MainLayout';

export default function NotFound() {
  const navigate = useNavigate();

  return (
    <MainLayout>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '60vh',
        }}
      >
        <Typography variant="h1" color="primary" sx={{ fontWeight: 'bold', fontSize: '5rem' }}>
          404
        </Typography>
        <Typography variant="h5" gutterBottom>
          Page Not Found
        </Typography>
        <Typography variant="body1" sx={{ mb: 4 }}>
          The page you're looking for doesn't exist or has been moved.
        </Typography>
        <Button variant="contained" onClick={() => navigate('/')}>
          Return to Home
        </Button>
      </Box>
    </MainLayout>
  );
}