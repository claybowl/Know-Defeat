import { extendTheme } from '@chakra-ui/react';

const theme = extendTheme({
  colors: {
    brand: {
      50: '#e5f4ff',
      100: '#c1daff',
      200: '#9bc1ff',
      300: '#75a7ff',
      400: '#508eff',
      500: '#3674e5',
      600: '#265bb3',
      700: '#174282',
      800: '#072852',
      900: '#001020',
    },
    success: {
      500: '#38A169', // Green for profits
    },
    danger: {
      500: '#E53E3E', // Red for losses
    },
  },
  fonts: {
    heading: 'Inter, system-ui, sans-serif',
    body: 'Inter, system-ui, sans-serif',
  },
  config: {
    initialColorMode: 'light',
    useSystemColorMode: false,
  },
  styles: {
    global: (props) => ({
      body: {
        bg: props.colorMode === 'dark' ? 'gray.900' : 'gray.50',
      },
    }),
  },
  components: {
    Button: {
      defaultProps: {
        colorScheme: 'brand',
      },
    },
    Card: {
      baseStyle: {
        container: {
          borderRadius: 'lg',
          boxShadow: 'md',
        },
      },
    },
  },
});

export default theme;