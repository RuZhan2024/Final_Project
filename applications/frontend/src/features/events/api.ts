import { apiRequest, type ApiRequestOptions } from "../../lib/apiClient";
import type {
  EventReviewUpdate,
  EventSkeletonClipResponse,
  EventsResponse,
  EventsSummaryResponse,
  EventStatus,
} from "./types";

export interface FetchEventsOptions extends Pick<ApiRequestOptions, "signal"> {
  residentId?: number;
  limit?: number;
}

export async function fetchEvents(
  apiBase: string,
  { residentId = 1, limit = 500, signal }: FetchEventsOptions = {}
): Promise<EventsResponse> {
  return await apiRequest<EventsResponse>(apiBase, `/api/events?resident_id=${residentId}&limit=${limit}`, {
    signal,
  });
}

export interface FetchEventsSummaryOptions extends Pick<ApiRequestOptions, "signal"> {
  residentId?: number;
}

export async function fetchEventsSummary(
  apiBase: string,
  { residentId = 1, signal }: FetchEventsSummaryOptions = {}
): Promise<EventsSummaryResponse> {
  return await apiRequest<EventsSummaryResponse>(apiBase, `/api/events/summary?resident_id=${residentId}`, {
    signal,
  });
}

export async function updateEventStatus(
  apiBase: string,
  eventId: number | string,
  status: EventStatus
): Promise<EventReviewUpdate> {
  return await apiRequest<EventReviewUpdate>(apiBase, `/api/events/${eventId}/status`, {
    method: "PUT",
    body: { status },
  });
}

export interface FetchEventSkeletonClipOptions extends Pick<ApiRequestOptions, "signal"> {
  residentId?: number;
}

export async function fetchEventSkeletonClip(
  apiBase: string,
  eventId: number | string,
  { residentId = 1, signal }: FetchEventSkeletonClipOptions = {}
): Promise<EventSkeletonClipResponse> {
  return await apiRequest<EventSkeletonClipResponse>(
    apiBase,
    `/api/events/${eventId}/skeleton_clip?resident_id=${encodeURIComponent(residentId)}`,
    { signal }
  );
}
