import { Box, Container, CssBaseline } from '@mui/material';
import Header from './Header';
import Footer from './Footer';

interface MainLayoutProps {
  children: React.ReactNode;
  maxWidth?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | false;
}

export default function MainLayout({ 
  children, 
  maxWidth = 'lg'
}: MainLayoutProps) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
      }}
    >
      <CssBaseline />
      <Header />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          py: 3,
        }}
      >
        <Container maxWidth={maxWidth}>
          {children}
        </Container>
      </Box>
      <Footer />
    </Box>
  );
}