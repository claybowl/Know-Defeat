import { AppBar, Toolbar, Typography, Button, Box, Container } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';

export default function Header() {
  return (
    <AppBar position="static" color="primary">
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <Typography
            variant="h6"
            noWrap
            component={RouterLink}
            to="/"
            sx={{
              mr: 2,
              display: { xs: 'none', md: 'flex' },
              fontWeight: 700,
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            Know Defeat Trading
          </Typography>

          <Box sx={{ flexGrow: 1, display: 'flex' }}>
            <Button
              component={RouterLink}
              to="/dashboard"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              Dashboard
            </Button>
            <Button
              component={RouterLink}
              to="/bots"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              Bots
            </Button>
            <Button
              component={RouterLink}
              to="/trades"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              Trades
            </Button>
            <Button
              component={RouterLink}
              to="/metrics"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              Metrics
            </Button>
            <Button
              component={RouterLink}
              to="/allocation"
              sx={{ my: 2, color: 'white', display: 'block' }}
            >
              Allocation
            </Button>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
}