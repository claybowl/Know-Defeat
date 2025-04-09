import { Box, Container, Flex } from '@chakra-ui/react';
import Header from './Header';
import Footer from './Footer';

interface MainLayoutProps {
  children: React.ReactNode;
  maxWidth?: string;
  withoutFooter?: boolean;
}

export default function MainLayout({ 
  children, 
  maxWidth = 'container.xl',
  withoutFooter = false
}: MainLayoutProps) {
  return (
    <Flex
      direction="column"
      minH="100vh"
    >
      <Header />
      <Box flex="1" as="main">
        <Container maxW={maxWidth} py={8} px={4}>
          {children}
        </Container>
      </Box>
      {!withoutFooter && <Footer />}
    </Flex>
  );
}