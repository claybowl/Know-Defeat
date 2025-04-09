import {
  Box,
  Container,
  Stack,
  Text,
  Link,
  useColorModeValue,
  Flex,
  Image,
} from '@chakra-ui/react';
import { Link as RemixLink } from '@remix-run/react';

export default function Footer() {
  return (
    <Box
      bg={useColorModeValue('gray.50', 'gray.900')}
      color={useColorModeValue('gray.700', 'gray.200')}
      borderTopWidth={1}
      borderStyle={'solid'}
      borderColor={useColorModeValue('gray.200', 'gray.700')}
    >
      <Container
        as={Stack}
        maxW={'container.xl'}
        py={4}
        spacing={4}
        justify={'space-between'}
        align={'center'}
      >
        <Flex align="center">
          <Image 
            src="/assets/know_defeat_gpt.png" 
            alt="Know Defeat Logo" 
            h="30px" 
            w="30px" 
            borderRadius="full" 
            mr={2}
          />
          <Text fontWeight="bold">Know Defeat Trading System</Text>
        </Flex>
        <Stack direction={'row'} spacing={6}>
          <Link as={RemixLink} to={'/'}>Home</Link>
          <Link as={RemixLink} to={'/dashboard'}>Dashboard</Link>
          <Link as={RemixLink} to={'/bots'}>Bots</Link>
          <Link as={RemixLink} to={'/trades'}>Trades</Link>
          <Link as={RemixLink} to={'/metrics'}>Metrics</Link>
        </Stack>
        <Text>© {new Date().getFullYear()} Know Defeat. All rights reserved</Text>
      </Container>
    </Box>
  );
}