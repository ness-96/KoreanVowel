create extension if not exists pgcrypto;

create table if not exists public.vowel_batches (
  batch_id text primary key,
  client_request_id text not null unique,
  created_at timestamptz not null default now(),
  created_by text not null default 'manual',
  record_count integer not null default 0
);

create table if not exists public.vowel_tokens (
  id text primary key,
  batch_id text not null references public.vowel_batches(batch_id) on delete restrict,
  speaker_id text not null,
  region text not null,
  country text,
  nationality text,
  gender text not null check (gender in ('female', 'male')),
  age text not null,
  vowel text not null check (vowel in ('ㅣ','ㅔ','ㅐ','ㅏ','ㅓ','ㅡ','ㅗ','ㅜ')),
  f1 double precision not null check (f1 > 0 and f1 < 5000),
  f2 double precision not null check (f2 > 0 and f2 < 5000),
  token_no integer not null check (token_no > 0),
  source text not null default 'manual',
  created_at timestamptz not null default now(),
  deleted_at timestamptz
);

create index if not exists vowel_tokens_active_idx on public.vowel_tokens (deleted_at, created_at);
create index if not exists vowel_tokens_speaker_idx on public.vowel_tokens (speaker_id, vowel, token_no);

alter table public.vowel_batches enable row level security;
alter table public.vowel_tokens enable row level security;
revoke all on public.vowel_batches from anon, authenticated;
revoke all on public.vowel_tokens from anon, authenticated;

create or replace function public.vowel_token_json()
returns jsonb
language sql
stable
security definer
set search_path = public
as $$
  select coalesce(jsonb_agg(jsonb_build_object(
    'id', v.id,
    'speakerId', v.speaker_id,
    'region', v.region,
    'country', coalesce(v.country, ''),
    'nationality', coalesce(v.nationality, ''),
    'gender', v.gender,
    'age', v.age,
    'vowel', v.vowel,
    'f1', v.f1,
    'f2', v.f2,
    'tokenNo', v.token_no,
    'source', v.source,
    'batchId', v.batch_id,
    'createdAt', v.created_at
  ) order by v.created_at, v.id), '[]'::jsonb)
  from public.vowel_tokens v
  where v.deleted_at is null;
$$;

create or replace function public.append_vowel_batch(
  p_batch_id text,
  p_client_request_id text,
  p_records jsonb
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  existing_batch text;
  inserted_count integer;
  rows_json jsonb;
begin
  if nullif(trim(p_batch_id), '') is null or nullif(trim(p_client_request_id), '') is null then
    raise exception 'batch_id and client_request_id are required';
  end if;
  if jsonb_typeof(p_records) <> 'array' or jsonb_array_length(p_records) = 0 then
    raise exception 'p_records must be a non-empty array';
  end if;

  select batch_id into existing_batch
  from public.vowel_batches
  where client_request_id = p_client_request_id;

  if existing_batch is not null then
    select public.vowel_token_json() into rows_json;
    return jsonb_build_object('ok', true, 'batchId', existing_batch, 'inserted', false, 'speakerData', rows_json, 'averageData', '[]'::jsonb);
  end if;

  insert into public.vowel_batches (batch_id, client_request_id)
  values (p_batch_id, p_client_request_id);

  insert into public.vowel_tokens (
    id, batch_id, speaker_id, region, country, nationality, gender, age,
    vowel, f1, f2, token_no, source
  )
  select r.id, p_batch_id, r.speaker_id, r.region, nullif(r.country, ''), nullif(r.nationality, ''),
         r.gender, r.age, r.vowel, r.f1, r.f2, r.token_no, coalesce(nullif(r.source, ''), 'manual')
  from jsonb_to_recordset(p_records) as r(
    id text,
    speaker_id text,
    region text,
    country text,
    nationality text,
    gender text,
    age text,
    vowel text,
    f1 double precision,
    f2 double precision,
    token_no integer,
    source text
  )
  on conflict (id) do nothing;

  get diagnostics inserted_count = row_count;
  update public.vowel_batches set record_count = inserted_count where batch_id = p_batch_id;
  select public.vowel_token_json() into rows_json;
  return jsonb_build_object('ok', true, 'batchId', p_batch_id, 'inserted', true, 'insertedCount', inserted_count, 'speakerData', rows_json, 'averageData', '[]'::jsonb);
end;
$$;

create or replace function public.delete_vowel_tokens(p_ids text[], p_all boolean default false)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  deleted_count integer;
  rows_json jsonb;
begin
  if p_all then
    update public.vowel_tokens set deleted_at = now() where deleted_at is null;
  elsif coalesce(array_length(p_ids, 1), 0) > 0 then
    update public.vowel_tokens set deleted_at = now() where id = any(p_ids) and deleted_at is null;
  end if;
  get diagnostics deleted_count = row_count;
  select public.vowel_token_json() into rows_json;
  return jsonb_build_object('ok', true, 'deletedCount', deleted_count, 'speakerData', rows_json, 'averageData', '[]'::jsonb);
end;
$$;

revoke all on function public.vowel_token_json() from public, anon, authenticated;
revoke all on function public.append_vowel_batch(text, text, jsonb) from public, anon, authenticated;
revoke all on function public.delete_vowel_tokens(text[], boolean) from public, anon, authenticated;
grant execute on function public.vowel_token_json() to service_role;
grant execute on function public.append_vowel_batch(text, text, jsonb) to service_role;
grant execute on function public.delete_vowel_tokens(text[], boolean) to service_role;

grant select on public.vowel_batches to service_role;
grant select on public.vowel_tokens to service_role;
