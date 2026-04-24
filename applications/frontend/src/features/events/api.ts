/**
 * Events API helpers.
 *
 * These functions keep the review UI aligned with the backend event list,
 * summary, review-status, and skeleton-clip contracts.
 */
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
  /** Read the review-event list using the legacy resident/limit query contract. */
  // The events page still uses the historical resident/limit query shape for compatibility.
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
  /** Read the lightweight event summary used by dashboard and events pages. */
  return await apiRequest<EventsSummaryResponse>(apiBase, `/api/events/summary?resident_id=${residentId}`, {
    signal,
  });
}

export async function updateEventStatus(
  apiBase: string,
  eventId: number | string,
  status: EventStatus
): Promise<EventReviewUpdate> {
  /** Persist one reviewer-selected event status. */
  // Review status updates always send the normalized enum string expected by the backend.
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
  /** Load the stored skeleton clip payload for one reviewed event. */
  // resident_id stays in the query for parity with the backend route contract and older tests.
  return await apiRequest<EventSkeletonClipResponse>(
    apiBase,
    `/api/events/${eventId}/skeleton_clip?resident_id=${encodeURIComponent(residentId)}`,
    { signal }
  );
}
