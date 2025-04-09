import axios from 'axios';
import { Bot, BotDetail, Trade, BotMetrics, DashboardData, AllocationData } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Bots
export const getAllBots = async (): Promise<Bot[]> => {
  const response = await api.get<Bot[]>('/bots');
  return response.data;
};

export const getBotById = async (id: string | number): Promise<BotDetail> => {
  const response = await api.get<BotDetail>(`/bots/${id}`);
  return response.data;
};

// Trades
export const getAllTrades = async (limit = 100): Promise<Trade[]> => {
  const response = await api.get<Trade[]>(`/trades?limit=${limit}`);
  return response.data;
};

export const getOpenTrades = async (): Promise<Trade[]> => {
  const response = await api.get<Trade[]>('/trades/open');
  return response.data;
};

// Metrics
export const getBotMetrics = async (): Promise<BotMetrics[]> => {
  const response = await api.get<BotMetrics[]>('/metrics');
  return response.data;
};

// Dashboard
export const getDashboardData = async (): Promise<DashboardData> => {
  const response = await api.get<DashboardData>('/dashboard');
  return response.data;
};

// Allocation
export const getAllocationData = async (): Promise<AllocationData> => {
  const response = await api.get<AllocationData>('/allocation');
  return response.data;
};

export default {
  getAllBots,
  getBotById,
  getAllTrades,
  getOpenTrades,
  getBotMetrics,
  getDashboardData,
  getAllocationData,
};