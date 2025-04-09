import { Box, Typography, Container, Link } from '@mui/material';

export default function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        py: 3,
        px: 2,
        mt: 'auto',
        backgroundColor: (theme) =>
          theme.palette.mode === 'light'
            ? theme.palette.grey[200]
            : theme.palette.grey[800],
      }}
    >
      <Container maxWidth="lg">
        <Typography variant="body2" color="text.secondary" align="center">
          {'© '}
          <Link color="inherit" href="/">
            Know Defeat Trading System
          </Link>{' '}
          {new Date().getFullYear()}
          {' by Curve AI Solutions'}
        </Typography>
      </Container>
    </Box>
  );
}