import React, { useState } from 'react';
import {
    Box,
    Heading,
    VStack,
    useDisclosure,
    Spinner,
    Text,
    Alert,
    AlertIcon,
    // Potentially add other layout components like Flex, Input for filtering, etc.
} from '@chakra-ui/react';
import { BotDetailModal } from '../components/modals/BotDetailModal'; // Adjust path

// --- Placeholder for the actual Bot List Table --- 
// This component will be responsible for fetching bot list data 
// and rendering the table rows.
interface BotListTableProps {
    onViewDetails: (botId: number) => void;
    // Add props for filtering, sorting, pagination later
}

const BotListTablePlaceholder: React.FC<BotListTableProps> = ({ onViewDetails }) => {
    // In a real implementation, this component would use useSWR or similar
    // to fetch data from /api/metrics/bots and render a Chakra UI Table.
    // It would also handle pagination, sorting, and filtering controls.
    
    // Example of how a row click might trigger the detail view:
    const handleRowClick = (botId: number) => {
        onViewDetails(botId);
    };

    return (
        <Box borderWidth="1px" borderRadius="lg" p={4} shadow="sm">
            <Text mb={4} fontStyle="italic">Bot List Table Placeholder</Text>
            {/* Example clickable item to simulate row click */}
            <Box 
                as="button" 
                onClick={() => handleRowClick(1)} // Example: Trigger details for bot ID 1
                p={2} 
                borderWidth="1px" 
                borderRadius="md" 
                _hover={{ bg: "gray.100" }}
                mb={2}
                display="block"
            >
                Click to view details for Bot ID 1 (Example)
            </Box>
            <Box 
                as="button" 
                onClick={() => handleRowClick(123)} // Example: Trigger details for bot ID 123
                p={2} 
                borderWidth="1px" 
                borderRadius="md" 
                _hover={{ bg: "gray.100" }}
                display="block"
            >
                Click to view details for Bot ID 123 (Example)
            </Box>
             {/* Here you would map over fetched bot data and render Table rows */}
             {/* Each row would have an onClick handler calling handleRowClick(bot.bot_id) */}
        </Box>
    );
};
// --- End Placeholder --- 


const MetricsDashboard: React.FC = () => {
    const { isOpen, onOpen, onClose } = useDisclosure(); // Hook to manage modal state
    const [selectedBotId, setSelectedBotId] = useState<number | null>(null);

    // Callback function passed to the table to open the modal for a specific bot
    const handleViewDetails = (botId: number) => {
        setSelectedBotId(botId);
        onOpen(); // Open the modal
    };

    // Function to handle closing the modal
    const handleCloseModal = () => {
        onClose();
        setSelectedBotId(null); // Clear selected ID when closing
    };

    return (
        <Box p={{ base: 4, md: 6 }}>
            <VStack spacing={6} align="stretch">
                <Heading as="h1" size="xl">
                    Bot Metrics Dashboard
                </Heading>

                {/* Potential area for filters, search, or global stats - Add later */}
                {/* <Box> ... </Box> */}

                {/* Render the Bot List Table */}
                <BotListTablePlaceholder onViewDetails={handleViewDetails} />

            </VStack>

            {/* Render the Detail Modal (conditionally rendered is fine) */}
            {selectedBotId !== null && (
                <BotDetailModal
                    isOpen={isOpen}
                    onClose={handleCloseModal}
                    botId={selectedBotId}
                />
            )}
        </Box>
    );
};

export default MetricsDashboard; 